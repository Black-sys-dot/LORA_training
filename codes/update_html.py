html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LORA LAB - Gojo</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Caveat:wght@700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0d1117; --bg-panel: #161b22; --bg-input: #010409;
            --accent-blue: #58a6ff; --accent-glow: rgba(88, 166, 255, 0.4);
            --text-main: #c9d1d9; --text-muted: #8b949e; --border-color: #30363d;
        }
        body { margin: 0; padding: 0; background-color: var(--bg-dark); color: var(--text-main); font-family: 'Inter', sans-serif; display: flex; height: 100vh; overflow: hidden; }
        
        .sidebar-left { width: 240px; background-color: var(--bg-panel); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; padding: 1.5rem 1rem; }
        .logo { font-size: 1.2rem; font-weight: 600; color: white; letter-spacing: 1px; margin-bottom: 2rem; }
        .logo span { color: var(--text-muted); font-size: 0.7rem; font-weight: 300; display: block; letter-spacing: 0; }
        
        .nav-item { padding: 0.75rem 1rem; margin-bottom: 0.5rem; border-radius: 6px; cursor: pointer; color: var(--text-muted); font-weight: 600; transition: all 0.2s; }
        .nav-item.active { background-color: rgba(88, 166, 255, 0.1); color: var(--accent-blue); }
        .nav-item:hover:not(.active) { background-color: var(--bg-input); color: white; }
        
        .sidebar-quote { margin-top: auto; padding-top: 2rem; border-top: 1px solid var(--border-color); font-size: 0.85rem; color: var(--text-muted); font-style: italic; }

        .main-content { flex: 1; display: flex; flex-direction: column; padding: 2rem 3rem; overflow-y: auto; }
        .header-banner { margin-bottom: 2rem; }
        
        .control-panel { background-color: var(--bg-panel); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }
        .prompt-area label { display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; font-weight: 600; }
        .prompt-area textarea { width: 100%; background-color: var(--bg-input); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; color: white; font-family: 'Inter', sans-serif; font-size: 0.95rem; resize: vertical; min-height: 80px; box-sizing: border-box; margin-bottom: 1rem; }
        .prompt-area textarea:focus { outline: none; border-color: var(--accent-blue); box-shadow: 0 0 0 2px var(--accent-glow); }
        
        .settings-row { display: flex; gap: 1.5rem; align-items: flex-end; }
        .setting-group { flex: 1; display: flex; flex-direction: column; }
        .setting-group label { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.4rem; font-weight: 600;}
        .setting-group input, .setting-group select { background-color: var(--bg-input); border: 1px solid var(--border-color); border-radius: 6px; padding: 0.6rem; color: white; font-size: 0.9rem; }
        
        .generate-btn { background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 8px; padding: 0.8rem 2rem; font-weight: 600; font-size: 1rem; cursor: pointer; transition: all 0.2s; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4); height: 42px; display: flex; align-items: center; justify-content: center; }
        .generate-btn:hover { box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6); }
        .generate-btn:disabled { background: #30363d; color: #8b949e; cursor: not-allowed; box-shadow: none; }

        .quick-prompts { display: flex; gap: 10px; margin-top: 1rem; flex-wrap: wrap; }
        .quick-prompt-pill { background-color: var(--bg-input); border: 1px solid var(--border-color); border-radius: 20px; padding: 0.4rem 1rem; font-size: 0.8rem; cursor: pointer; color: var(--text-muted); transition: all 0.2s; }
        .quick-prompt-pill:hover { border-color: var(--accent-blue); color: var(--accent-blue); }

        .results-area { display: flex; gap: 20px; width: 100%; min-height: 400px; margin-bottom: 1rem; }
        .image-box { flex: 1; background-color: var(--bg-input); border: 1px solid var(--border-color); border-radius: 12px; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; }
        .image-box img { width: 100%; height: 100%; object-fit: cover; display: none; }
        .image-badge { position: absolute; top: 15px; left: 15px; background-color: rgba(0,0,0,0.7); backdrop-filter: blur(4px); padding: 0.4rem 1rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.5px; border: 1px solid rgba(255,255,255,0.1); }
        
        .badge-base { color: #ff7b72; }
        .badge-lora { color: var(--accent-blue); }
        .loader { border: 3px solid var(--border-color); border-top: 3px solid var(--accent-blue); border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; display: none; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* LIBRARY GRID */
        .library-grid { display: flex; flex-direction: column; gap: 2rem; padding-bottom: 2rem; }
        .library-item { background-color: var(--bg-panel); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.5rem; position: relative; }
        .library-pair { display: flex; gap: 20px; margin-top: 10px; }
        .library-pair img { flex: 1; width: 100%; max-height: 600px; object-fit: contain; border-radius: 8px; border: 1px solid var(--border-color); }
        .library-title { font-size: 1.1rem; color: white; font-weight: 600; margin-bottom: 0.5rem; }
        .library-prompt { font-size: 0.85rem; color: var(--text-muted); font-style: italic; }
        
        .action-btn { background-color: var(--bg-input); border: 1px solid var(--border-color); color: var(--text-muted); padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: 0.2s; }
        .action-btn:hover { color: white; border-color: white; }
        .fav-btn { position: absolute; top: 1.5rem; right: 1.5rem; font-size: 1.5rem; cursor: pointer; background: none; border: none; color: var(--text-muted); transition: 0.2s; }
        .fav-btn:hover, .fav-btn.favorited { color: #f1c40f; }

    </style>
</head>
<body>

    <!-- LEFT SIDEBAR -->
    <div class="sidebar-left">
        <div class="logo">LORA LAB<span>Adapters Change Perspective</span></div>
        <div class="nav-item active" id="nav-generate" onclick="switchTab('generate')">⬟ Generate</div>
        <div class="nav-item" id="nav-library" onclick="switchTab('library')">📚 Library</div>
        <div class="nav-item" id="nav-favorites" onclick="switchTab('favorites')">⭐ Favorites</div>
        
        <div class="sidebar-quote">
            "It's not magic.<br>It's just different weights."<br><br>
            <span style="font-family: 'Caveat', cursive; font-size: 1.5rem; color: white;">Satoru Gojo</span>
        </div>
    </div>

    <!-- MAIN CONTENT -->
    <div class="main-content">
        <div class="header-banner">
            <img src="stripe.png" alt="Same Prompt. Different Worlds." style="width: 100%; max-height: 120px; object-fit: cover; border-radius: 12px; border: 1px solid var(--border-color); box-shadow: 0 4px 15px rgba(0,0,0,0.5);" />
        </div>

        <!-- GENERATE TAB -->
        <div id="tab-generate">
            <div class="control-panel">
                <div class="prompt-area">
                    <label>Action / Environment Prompt</label>
                    <textarea id="prompt" placeholder="e.g. drinking coffee, sitting on a bench, walking in the rain...">eating a sweet crepe on a bustling Harajuku street during sunset.</textarea>
                </div>
                
                <div class="settings-row">
                    <div class="setting-group"><label>Base Model</label><select disabled><option>FLUX.1-dev</option></select></div>
                    <div class="setting-group"><label>Adapter</label><select disabled><option>Gojo_v1 (LoRA)</option></select></div>
                    <div class="setting-group"><label>Seed</label><input type="number" id="seed" value="1000" /></div>
                    <div class="setting-group" style="display:none;"><label>Token</label><input type="password" id="token" value="YOUR_TOKEN_HERE" /></div>
                    <button class="generate-btn" id="generateBtn" onclick="generateImage()">✨ Generate (2)</button>
                </div>

                <div class="quick-prompts">
                    <div class="quick-prompt-pill" onclick="setPrompt('eating a sweet crepe on a bustling Harajuku street during sunset.')">Crepe in Harajuku</div>
                    <div class="quick-prompt-pill" onclick="setPrompt('floating cross-legged in the sky above the clouds, glowing with blue cursed energy.')">Floating in Sky</div>
                    <div class="quick-prompt-pill" onclick="setPrompt('adjusting his black blindfold while fighting a massive cursed spirit in a dark subway station.')">Subway Fight</div>
                    <div class="quick-prompt-pill" onclick="setPrompt('sitting casually on a destroyed concrete pillar, rubble floating around him.')">Rubble Pillar</div>
                    <div class="quick-prompt-pill" onclick="setPrompt('teaching a class at Jujutsu High, standing in front of a chalkboard.')">Jujutsu High</div>
                </div>
            </div>

            <div class="results-area">
                <div class="image-box">
                    <div class="image-badge badge-base">⛶ BASE (No LoRA)</div>
                    <div class="loader" id="loader-base"></div>
                    <img id="baseImage" alt="Base Result" />
                </div>
                <div class="image-box">
                    <div class="image-badge badge-lora">✦ WITH LoRA</div>
                    <div class="loader" id="loader-lora"></div>
                    <img id="loraImage" alt="LoRA Result" />
                </div>
            </div>
            
            <div style="text-align: right;">
                <button class="action-btn" id="saveLibraryBtn" onclick="addCurrentToLibrary()" style="display:none;">📥 Add to Library</button>
            </div>
        </div>

        <!-- LIBRARY TAB -->
        <div id="tab-library" style="display:none;">
            <h2 style="color: white; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; margin-top:0;">Global Library</h2>
            <div class="library-grid" id="library-grid">
                <!-- Pre-populated from local directory -->
            </div>
        </div>

        <!-- FAVORITES TAB -->
        <div id="tab-favorites" style="display:none;">
            <h2 style="color: white; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; margin-top:0;">Favorites</h2>
            <div class="library-grid" id="favorites-grid">
                <!-- Favorites go here -->
            </div>
        </div>
    </div>

    <script>
        const API_URL = "YOUR_API_URL_HERE";
        let currentGeneration = null;
        let globalLibrary = [
            { id: 1, title: "Tea Garden", prompt: "Satoru Gojo drinking tea in a garden.", base: "../results/comparison2/comparison2_1_base.png", lora: "../results/comparison2/comparison2_1_lora.png", fav: true },
            { id: 2, title: "Rainy City Street", prompt: "Satoru Gojo, wearing a casual hoodie, standing in a rainy city street.", base: "../results/comparison2/comparison2_2_base.png", lora: "../results/comparison2/comparison2_2_lora.png", fav: false },
            { id: 3, title: "Crumbling Temple", prompt: "Satoru Gojo, glowing purple energy around his hands, levitating above a crumbling temple.", base: "../results/comparison2/comparison2_3_base.png", lora: "../results/comparison2/comparison2_3_lora.png", fav: true },
            { id: 4, title: "Purple Attack", prompt: "he is using purple attack with both hands", base: "../results/comparison2/comparison2_4_base.png", lora: "../results/comparison2/comparison2_4_lora.png", fav: true }
        ];

        function switchTab(tab) {
            document.getElementById('nav-generate').classList.remove('active');
            document.getElementById('nav-library').classList.remove('active');
            document.getElementById('nav-favorites').classList.remove('active');
            document.getElementById('nav-' + tab).classList.add('active');
            
            document.getElementById('tab-generate').style.display = 'none';
            document.getElementById('tab-library').style.display = 'none';
            document.getElementById('tab-favorites').style.display = 'none';
            document.getElementById('tab-' + tab).style.display = 'block';
            
            if(tab === 'library' || tab === 'favorites') {
                renderLibrary();
            }
        }

        function setPrompt(text) { document.getElementById('prompt').value = text; }

        function toggleFavorite(id) {
            const item = globalLibrary.find(i => i.id === id);
            if(item) { item.fav = !item.fav; renderLibrary(); }
        }

        function renderLibrary() {
            const libGrid = document.getElementById('library-grid');
            const favGrid = document.getElementById('favorites-grid');
            libGrid.innerHTML = ''; favGrid.innerHTML = '';
            
            globalLibrary.forEach(item => {
                const html = `
                    <div class="library-item">
                        <button class="fav-btn ${item.fav ? 'favorited' : ''}" onclick="toggleFavorite(${item.id})">★</button>
                        <div class="library-title">${item.title}</div>
                        <div class="library-prompt">"${item.prompt}"</div>
                        <div class="library-pair">
                            <div style="flex:1; position:relative;">
                                <div class="image-badge badge-base">⛶ BASE (No LoRA)</div>
                                <img src="${item.base}">
                            </div>
                            <div style="flex:1; position:relative;">
                                <div class="image-badge badge-lora">✦ WITH LoRA</div>
                                <img src="${item.lora}">
                            </div>
                        </div>
                    </div>
                `;
                libGrid.innerHTML += html;
                if(item.fav) { favGrid.innerHTML += html; }
            });
        }

        function addCurrentToLibrary() {
            if(!currentGeneration) return;
            globalLibrary.unshift({
                id: Date.now(),
                title: "Live Generation",
                prompt: currentGeneration.prompt,
                base: currentGeneration.base,
                lora: currentGeneration.lora,
                fav: false
            });
            document.getElementById('saveLibraryBtn').innerText = "✓ Added to Library!";
            document.getElementById('saveLibraryBtn').disabled = true;
            renderLibrary();
        }

        async function generateImage() {
            const prompt = document.getElementById("prompt").value;
            const seed = parseInt(document.getElementById("seed").value);
            const token = document.getElementById("token").value;
            const btn = document.getElementById("generateBtn");
            const baseImg = document.getElementById("baseImage");
            const loraImg = document.getElementById("loraImage");
            const loaderBase = document.getElementById("loader-base");
            const loaderLora = document.getElementById("loader-lora");
            const saveBtn = document.getElementById("saveLibraryBtn");

            btn.disabled = true; btn.innerText = "Triggering GPUs...";
            baseImg.style.display = "none"; loraImg.style.display = "none"; saveBtn.style.display = "none";
            loaderBase.style.display = "block"; loaderLora.style.display = "block";

            async function fetchModelImage(useLora) {
                const res = await fetch(API_URL, {
                    method: "POST", headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
                    body: JSON.stringify({ prompt: prompt, seed: seed, use_lora: useLora })
                });
                if (!res.ok) throw new Error(`Error ${res.status}`);
                const blob = await res.blob();
                return URL.createObjectURL(blob);
            }

            try {
                const [baseUrl, loraUrl] = await Promise.all([ fetchModelImage(false), fetchModelImage(true) ]);
                baseImg.src = baseUrl; loraImg.src = loraUrl;
                loaderBase.style.display = "none"; loaderLora.style.display = "none";
                baseImg.style.display = "block"; loraImg.style.display = "block";
                
                // Save state for library
                currentGeneration = { prompt: prompt, base: baseUrl, lora: loraUrl };
                saveBtn.style.display = "inline-block";
                saveBtn.innerText = "📥 Add to Library";
                saveBtn.disabled = false;
            } catch (e) {
                alert("Failed: " + e.message);
                loaderBase.style.display = "none"; loaderLora.style.display = "none";
            } finally {
                btn.disabled = false; btn.innerText = "✨ Generate (2)";
            }
        }

        // Init
        renderLibrary();
    </script>
</body>
</html>
"""
with open("../frontend/index.html", "w") as f:
    f.write(html)
print("Updated frontend!")
