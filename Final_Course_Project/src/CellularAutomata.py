import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn

# ---------- 1. Datos ----------
train = pd.read_csv("train.csv")

X = train.drop(["id", "target"], axis=1).values.astype(np.float32)
y_str = train["target"].values

le = LabelEncoder()
y = le.fit_transform(y_str).astype(np.int64)
num_classes = len(le.classes_)

scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X).astype(np.float32)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# vecinos k-NN en train
k = 5
nn_tr = NearestNeighbors(n_neighbors=k, metric="euclidean")
nn_tr.fit(X_tr)
neighbors_tr = nn_tr.kneighbors(X_tr, return_distance=False)

nn_val = NearestNeighbors(n_neighbors=k, metric="euclidean")
nn_val.fit(X_val)
neighbors_val = nn_val.kneighbors(X_val, return_distance=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

Xtr_t  = torch.from_numpy(X_tr).to(device)
Xval_t = torch.from_numpy(X_val).to(device)
ytr_t  = torch.from_numpy(y_tr).to(device)
yval_t = torch.from_numpy(y_val).to(device)
neighbors_tr_t  = torch.from_numpy(neighbors_tr).long().to(device)
neighbors_val_t = torch.from_numpy(neighbors_val).long().to(device)

# ---------- 2. Modelo CA ----------
class GraphCA(nn.Module):
    def __init__(self, in_dim, n_classes, T=3):
        super().__init__()
        self.T = T
        self.init_layer = nn.Linear(in_dim, n_classes)
        self.alpha = nn.Parameter(torch.tensor(0.8))
        self.beta  = nn.Parameter(torch.tensor(0.2))

    def forward(self, X, neighbors):
        logits = self.init_layer(X)           # (N,C)
        probs  = torch.softmax(logits, dim=1)

        eps = 1e-6
        for _ in range(self.T):
            # vecinos: (N,k,C)
            neigh_probs = probs[neighbors]
            neigh_mean  = neigh_probs.mean(dim=1)

            p_clip = probs.clamp(eps, 1-eps)
            n_clip = neigh_mean.clamp(eps, 1-eps)
            logit_self  = torch.log(p_clip) - torch.log(1-p_clip)
            logit_neigh = torch.log(n_clip) - torch.log(1-n_clip)

            mixed_logits = self.alpha * logit_self + self.beta * logit_neigh
            probs = torch.softmax(mixed_logits, dim=1)

        return probs

model = GraphCA(X_tr.shape[1], num_classes, T=3).to(device)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def multiclass_logloss_t(y_true, P):
    eps = 1e-15
    P = torch.clamp(P, eps, 1-eps)
    P = P / P.sum(dim=1, keepdim=True)
    return -torch.log(P[torch.arange(P.size(0)), y_true]).mean()

# ---------- 3. Entrenamiento sin batches ----------
for epoch in range(20):  # ajusta si hace falta
    model.train()
    P_tr = model(Xtr_t, neighbors_tr_t)
    logP_tr = torch.log(P_tr + 1e-15)
    loss = criterion(logP_tr, ytr_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        P_val = model(Xval_t, neighbors_val_t)
        val_loss = multiclass_logloss_t(yval_t, P_val).item()

print(f"Epoch {epoch+1:02d} | train NLL={loss.item():.4f} | val logloss={val_loss:.4f}")

# =========================================================
# 4. Entrenar con TODO el train y predecir test
# =========================================================

# Cargar train y test completos para la fase final
test = pd.read_csv("test.csv")
X_full = train.drop(["id", "target"], axis=1).values.astype(np.float32)
y_full = y  # ya lo teníamos codificado arriba
X_test = test.drop(["id"], axis=1).values.astype(np.float32)
ids_test = test["id"].values

# Reutilizar el escalador entrenado
X_full_scaled = scaler.transform(X_full).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# Recalcular vecinos sobre TODO el train
nn_all = NearestNeighbors(n_neighbors=k, metric="euclidean")
nn_all.fit(X_full_scaled)
neighbors_all = nn_all.kneighbors(X_full_scaled, return_distance=False)

# Vecinos para test con respecto a todo el train
nn_test = NearestNeighbors(n_neighbors=k, metric="euclidean")
nn_test.fit(X_full_scaled)
neighbors_test = nn_test.kneighbors(X_test_scaled, return_distance=False)

Xall_t = torch.from_numpy(X_full_scaled).to(device)
yall_t = torch.from_numpy(y_full).to(device)
neighbors_all_t = torch.from_numpy(neighbors_all).long().to(device)
Xtest_t = torch.from_numpy(X_test_scaled).to(device)
neighbors_test_t = torch.from_numpy(neighbors_test).long().to(device)

# Opcional: unas pocas epochs extra usando TODO el train
extra_epochs = 10
for epoch in range(extra_epochs):
    model.train()
    P_all = model(Xall_t, neighbors_all_t)
    logP_all = torch.log(P_all + 1e-15)
    loss_all = criterion(logP_all, yall_t)

    optimizer.zero_grad()
    loss_all.backward()
    optimizer.step()
print(f"[Full train] Epoch {epoch+1:02d} | NLL={loss_all.item():.4f}")

# Predicciones finales para test
model.eval()
with torch.no_grad():
    P_test = model(Xtest_t, neighbors_test_t).cpu().numpy()

# Recorte y renormalización estilo Kaggle
eps = 1e-15
P_test = np.clip(P_test, eps, 1 - eps)
P_test = P_test / P_test.sum(axis=1, keepdims=True)

# =========================================================
# 5. Crear submission.csv
# =========================================================
submission_df = pd.DataFrame(P_test, columns=le.classes_)
submission_df.insert(0, "id", ids_test)

prob_cols = le.classes_
submission_df[prob_cols] = submission_df[prob_cols].round(8)

submission_path = "submissionkaggle.csv"
submission_df.to_csv(submission_path, index=False)
print("Archivo generado:", submission_path)
