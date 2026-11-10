from flask import Flask, render_template_string
from flask_socketio import SocketIO
import subprocess
import re
import threading
import os
import sys

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Live LoRA Training Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; background: #121212; color: #fff; }
        .progress-container { width: 100%; background: #333; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }
        .progress-bar { width: 0%; height: 35px; background: #00d2ff; background: -webkit-linear-gradient(to right, #3a7bd5, #00d2ff); background: linear-gradient(to right, #3a7bd5, #00d2ff); text-align: center; line-height: 35px; color: white; font-weight: bold; transition: width 0.3s ease; }
        .logs { background: #1e1e1e; color: #a5d6a7; padding: 15px; height: 350px; border-radius: 8px; overflow-y: scroll; font-family: 'Courier New', Courier, monospace; white-space: pre-wrap; font-size: 14px; border: 1px solid #333; }
        .header { display: flex; justify-content: space-between; align-items: center; }
        #chart-container { width: 100%; height: 400px; background: #1e1e1e; padding: 20px; border-radius: 8px; box-sizing: border-box; margin-bottom: 20px; }
        h1, h3 { margin-top: 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>FLUX.1 [dev] LoRA Dashboard</h1>
        <h3 id="status">Status: Preparing...</h3>
    </div>
    
    <div class="progress-container">
        <div class="progress-bar" id="progress-bar">0%</div>
    </div>
    
    <div id="chart-container">
        <canvas id="lossChart"></canvas>
    </div>
    
    <h3>Live Logs</h3>
    <div class="logs" id="logs"></div>

    <script>
        var socket = io();
        var ctx = document.getElementById('lossChart').getContext('2d');
        
        Chart.defaults.color = '#fff';
        Chart.defaults.borderColor = '#444';
        
        var chart = new Chart(ctx, {
            type: 'line',
            data: { 
                labels: [], 
                datasets: [{ 
                    label: 'Training Loss', 
                    data: [], 
                    borderColor: '#ff6384', 
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.2 
                }] 
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false, 
                plugins: { legend: { display: false } },
                scales: { 
                    y: { beginAtZero: false },
                    x: { title: { display: true, text: 'Steps' } }
                } 
            }
        });

        socket.on('log', function(msg) {
            var logs = document.getElementById('logs');
            logs.innerHTML += msg.data + '\\n';
            logs.scrollTop = logs.scrollHeight;
        });

        socket.on('progress', function(msg) {
            var step = msg.step;
            var total = msg.total;
            var loss = msg.loss;
            var pct = Math.round((step / total) * 100);
            
            document.getElementById('progress-bar').style.width = pct + '%';
            document.getElementById('progress-bar').innerText = step + ' / ' + total + ' (' + pct + '%)';
            document.getElementById('status').innerText = 'Status: Training | Loss: ' + loss.toFixed(4);
            
            chart.data.labels.push(step);
            chart.data.datasets[0].data.push(loss);
            chart.update();
        });
        
        socket.on('status', function(msg) {
            document.getElementById('status').innerText = msg.data;
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

def run_modal_training():
    try:
        # Step 1: Create Volume
        socketio.emit('log', {'data': 'Creating Modal volume "flux-lora-volume"...'})
        os.system("modal volume create flux-lora-volume")
        
        # Step 2: Upload dataset
        socketio.emit('log', {'data': 'Uploading dataset to Modal volume...'})
        # Path assumes we are in the codes folder
        os.system("modal volume put flux-lora-volume ../dataset/gojo /dataset/gojo")
        socketio.emit('log', {'data': 'Dataset uploaded successfully.'})
        
        # Step 3: Run Training
        socketio.emit('log', {'data': 'Starting Modal training job...'})
        socketio.emit('status', {'data': 'Status: Initializing GPU...'})
        
        cmd = ["modal", "run", "train.py"]
        # Use unbuffered output and pipe both stdout and stderr
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Regex to catch ai-toolkit tqdm lines like:
        # " 10%|█         | 200/2000 [05:00<40:00, 1.50s/it, loss=0.105]"
        tqdm_regex = re.compile(r'(\d+)/(\d+) .*?loss=([\d\.]+)')
        
        # Read character by character because tqdm uses carriage returns (\\r) instead of newlines
        buffer = ""
        while True:
            char = process.stdout.read(1)
            if not char and process.poll() is not None:
                break
                
            if char in ('\\n', '\\r'):
                line = buffer.strip()
                buffer = ""
                
                if line:
                    # Emit log line
                    socketio.emit('log', {'data': line})
                    
                    # Try to parse progress
                    match = tqdm_regex.search(line)
                    if match:
                        step = int(match.group(1))
                        total = int(match.group(2))
                        loss = float(match.group(3))
                        socketio.emit('progress', {'step': step, 'total': total, 'loss': loss})
            else:
                buffer += char
                
        process.wait()
        
        if process.returncode == 0:
            socketio.emit('log', {'data': '\\n==== TRAINING COMPLETED SUCCESSFULLY ===='})
            socketio.emit('status', {'data': 'Status: Completed'})
            threading.Timer(5.0, lambda: os._exit(0)).start()
        else:
            socketio.emit('log', {'data': f'\\n==== TRAINING FAILED WITH CODE {process.returncode} ===='})
            socketio.emit('status', {'data': 'Status: Failed'})
            # We exit after a few seconds so the frontend can receive the failure message
            threading.Timer(5.0, lambda: os._exit(1)).start()
            
    except Exception as e:
        socketio.emit('log', {'data': f'Error: {str(e)}'})
        socketio.emit('status', {'data': 'Status: Error'})
        threading.Timer(5.0, lambda: os._exit(1)).start()

if __name__ == '__main__':
    # Start training in a background thread
    threading.Thread(target=run_modal_training).start()
    
    print("==================================================")
    print("Dashboard starting at http://localhost:5000")
    print("==================================================")
    
    # Run the Flask app
    socketio.run(app, host='0.0.0.0', port=5000, log_output=False)
