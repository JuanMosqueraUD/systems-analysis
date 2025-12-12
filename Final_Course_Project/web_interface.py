"""
Web Interface for Otto Classification Pipeline
Simple Flask application to run the pipeline and monitor logs in real-time
"""

import os
import subprocess
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
import threading
import queue
import time

app = Flask(__name__)

# Global variables for process management
current_process = None
log_queue = queue.Queue()
process_lock = threading.Lock()

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent

def tail_log_file(log_path, stop_event):
    """Tail log file and put lines into queue"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            # Go to end of file
            f.seek(0, 2)
            
            while not stop_event.is_set():
                line = f.readline()
                if line:
                    log_queue.put(line.rstrip('\n'))
                else:
                    time.sleep(0.1)
    except Exception as e:
        log_queue.put(f"Error reading log: {str(e)}")

def run_pipeline(command):
    """Run the pipeline command and capture output"""
    global current_process
    
    project_root = get_project_root()
    log_dir = project_root / 'outputs' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'pipeline.log'
    
    # Clear log queue
    while not log_queue.empty():
        log_queue.get()
    
    # Clear previous log file
    if log_file.exists():
        log_file.unlink()
    
    try:
        # Parse command
        cmd_parts = command.strip().split()
        
        # Build full command
        if cmd_parts[0] == 'python':
            full_command = cmd_parts
        else:
            full_command = ['python'] + cmd_parts
        
        log_queue.put(f"Starting command: {' '.join(full_command)}")
        log_queue.put("=" * 70)
        
        # Start log tailing in background
        stop_event = threading.Event()
        tail_thread = threading.Thread(target=tail_log_file, args=(log_file, stop_event))
        tail_thread.daemon = True
        tail_thread.start()
        
        # Run process
        with process_lock:
            current_process = subprocess.Popen(
                full_command,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
        
        # Stream output
        for line in current_process.stdout:
            log_queue.put(line.rstrip('\n'))
        
        # Wait for completion
        return_code = current_process.wait()
        
        # Stop log tailing
        stop_event.set()
        tail_thread.join(timeout=2)
        
        log_queue.put("=" * 70)
        if return_code == 0:
            log_queue.put("PIPELINE COMPLETED SUCCESSFULLY!")
            log_queue.put("Check outputs/submission.csv for results")
        else:
            log_queue.put(f"Pipeline finished with return code: {return_code}")
        
        return return_code
        
    except Exception as e:
        log_queue.put(f"ERROR: {str(e)}")
        return 1
    finally:
        with process_lock:
            current_process = None

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run():
    """Start pipeline execution"""
    data = request.json
    command = data.get('command', 'src/main.py')
    
    # Check if process is already running
    with process_lock:
        if current_process and current_process.poll() is None:
            return jsonify({'error': 'Pipeline is already running'}), 400
    
    # Start pipeline in background thread
    thread = threading.Thread(target=run_pipeline, args=(command,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/stream')
def stream():
    """Stream logs to client"""
    def generate():
        while True:
            try:
                # Get log line from queue with timeout
                line = log_queue.get(timeout=1)
                yield f"data: {json.dumps({'log': line})}\n\n"
            except queue.Empty:
                # Check if process is still running
                with process_lock:
                    if current_process is None or current_process.poll() is not None:
                        # Process finished, send completion signal
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        break
                # Send keepalive
                yield f"data: {json.dumps({'keepalive': True})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/status')
def status():
    """Get current status"""
    with process_lock:
        running = current_process is not None and current_process.poll() is None
    
    # Check for results
    project_root = get_project_root()
    results_file = project_root / 'outputs' / 'training_summary.json'
    submission_file = project_root / 'outputs' / 'submission.csv'
    
    results = None
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            pass
    
    return jsonify({
        'running': running,
        'has_results': results_file.exists(),
        'has_submission': submission_file.exists(),
        'results': results
    })

@app.route('/stop', methods=['POST'])
def stop():
    """Stop current execution"""
    with process_lock:
        if current_process and current_process.poll() is None:
            current_process.terminate()
            log_queue.put("=" * 70)
            log_queue.put("PIPELINE STOPPED BY USER")
            return jsonify({'status': 'stopped'})
    
    return jsonify({'error': 'No running process'}), 400

if __name__ == '__main__':
    print("=" * 70)
    print("Otto Classification Pipeline - Web Interface")
    print("=" * 70)
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
