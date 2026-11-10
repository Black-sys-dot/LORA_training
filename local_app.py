import http.server
import socketserver
import json
import os
import urllib.request
import time

PORT = 8001
LIBRARY_FILE = "frontend/library.json"

# Default library
default_library = [
    { "id": 1, "title": "Tea Garden", "prompt": "Satoru Gojo drinking tea in a garden.", "base": "../results/comparison2/comparison2_1_base.png", "lora": "../results/comparison2/comparison2_1_lora.png", "fav": True },
    { "id": 2, "title": "Rainy City Street", "prompt": "Satoru Gojo, wearing a casual hoodie, standing in a rainy city street.", "base": "../results/comparison2/comparison2_2_base.png", "lora": "../results/comparison2/comparison2_2_lora.png", "fav": False },
    { "id": 3, "title": "Crumbling Temple", "prompt": "Satoru Gojo, glowing purple energy around his hands, levitating above a crumbling temple.", "base": "../results/comparison2/comparison2_3_base.png", "lora": "../results/comparison2/comparison2_3_lora.png", "fav": True },
    { "id": 4, "title": "Purple Attack", "prompt": "he is using purple attack with both hands", "base": "../results/comparison2/comparison2_4_base.png", "lora": "../results/comparison2/comparison2_4_lora.png", "fav": True },
    { "id": 5, "title": "Portrait Test", "prompt": "Gojo portrait", "base": "../results/comparison/comparison_1_base.png", "lora": "../results/comparison/comparison_1_lora.png", "fav": False },
    { "id": 6, "title": "Action Test", "prompt": "Gojo action pose", "base": "../results/comparison/comparison_2_base.png", "lora": "../results/comparison/comparison_2_lora.png", "fav": False },
    { "id": 7, "title": "Cinematic Test", "prompt": "Gojo cinematic lighting", "base": "../results/comparison/comparison_3_base.png", "lora": "../results/comparison/comparison_3_lora.png", "fav": False },
    { "id": 8, "title": "Crepe in Harajuku", "prompt": "eating a sweet crepe on a bustling Harajuku street during sunset.", "base": "../results/comparison/test_crepe_s1000_base.png", "lora": "../results/comparison/test_crepe_s1000_lora.png", "fav": True },
    { "id": 9, "title": "Floating in Sky", "prompt": "floating cross-legged in the sky above the clouds, glowing with blue cursed energy.", "base": "../results/comparison/test_floating_s1000_base.png", "lora": "../results/comparison/test_floating_s1000_lora.png", "fav": True },
    { "id": 10, "title": "Subway Fight", "prompt": "adjusting his black blindfold while fighting a massive cursed spirit in a dark subway station.", "base": "../results/comparison/test_subway_s1000_base.png", "lora": "../results/comparison/test_subway_s1000_lora.png", "fav": True },
    { "id": 11, "title": "Rubble Pillar", "prompt": "sitting casually on a destroyed concrete pillar, rubble floating around him.", "base": "../results/comparison/test_rubble_s1000_base.png", "lora": "../results/comparison/test_rubble_s1000_lora.png", "fav": True },
    { "id": 12, "title": "Jujutsu High", "prompt": "teaching a class at Jujutsu High, standing in front of a chalkboard.", "base": "../results/comparison/test_teaching_s1000_base.png", "lora": "../results/comparison/test_teaching_s1000_lora.png", "fav": True }
]

if not os.path.exists(LIBRARY_FILE):
    with open(LIBRARY_FILE, "w") as f:
        json.dump(default_library, f)

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/frontend/")
            self.end_headers()
            return
        elif self.path == "/api/library":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            with open(LIBRARY_FILE, "rb") as f:
                self.wfile.write(f.read())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/library":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            with open(LIBRARY_FILE, "wb") as f:
                f.write(post_data)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "saved"}')

        elif self.path == "/api/download":
            # Saves blob urls locally by fetching them! Wait, python can't fetch a browser 'blob:' url.
            # We will instead have the JS send base64 data to this endpoint.
            import base64
            content_length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(content_length))
            
            timestamp = int(time.time())
            base_filename = f"results/comparison2/live_{timestamp}_base.png"
            lora_filename = f"results/comparison2/live_{timestamp}_lora.png"
            
            base_data = base64.b64decode(data['base64_base'].split(',')[1])
            lora_data = base64.b64decode(data['base64_lora'].split(',')[1])
            
            with open(base_filename, "wb") as f: f.write(base_data)
            with open(lora_filename, "wb") as f: f.write(lora_data)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "base_url": f"../{base_filename}",
                "lora_url": f"../{lora_filename}"
            }).encode())

with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"Custom persistence server running at port {PORT}")
    httpd.serve_forever()
