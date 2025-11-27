# Workshop 2 - System Design

This folder contains the work for Workshop 2 of the Systems Analysis course, focusing on system design, architecture definition, and technical stack selection.

## Objectives

Based on the system analysis from Workshop 1, this workshop defines:

- System requirements (functional and non-functional)
- Modular system architecture
- Technical stack and implementation tools
- Strategies to address chaos and sensitivity
- Data flow and component integration

## System Requirements

### Design Requirements
- Performance: Process 200,000+ records efficiently
- Reliability: Stable and reproducible results
- Accuracy and Calibration: Well-calibrated probability distributions
- Scalability: Support incremental data updates
- Traceability: Documented and reproducible processes

### User-Centric Requirements
- Ease of use through configuration files
- Interpretability with visualization tools
- Security and data integrity
- System stability with feedback control
- Maintainability through modular architecture

## System Architecture

The proposed architecture consists of modular components:

1. **Product Input**: Entry point for raw product data
2. **Data Processing**: Cleaning, normalization, and transformation
3. **Feature Engineering**: Feature extraction and construction
4. **Classification Engine**: Core unsupervised learning algorithms
5. **Analytics and Reporting**: Performance analysis and visualization
6. **Output Target File**: Structured classification results

## Technical Stack

- **Python**: Main programming environment
- **NumPy**: Numerical computation and matrix operations
- **Pandas**: Data management and transformation
- **Scikit-learn**: Machine learning algorithms and evaluation
- **Matplotlib**: Visualization and reporting

## Addressing Sensitivity and Chaos

- Feedback loops between modules for stability
- Structured data validation and modular control
- System adaptability and learning mechanisms
- Real-time performance monitoring

## Team Members

- Juan Diego Lozada 20222020014
- Juan Pablo Mosquera 20221020026
- María Alejandra Ortiz Sánchez 20242020223
- Jeison Felipe Cuenca 20242020043

## Deliverables

- System design document (main.tex)
- Architecture diagram
- Technical stack specification
- Module integration plan