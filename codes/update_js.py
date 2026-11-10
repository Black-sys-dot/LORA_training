html = open("../frontend/index.html").read()

# Replace the script section
script_start = html.find("<script>")
script_end = html.find("</script>") + 9

new_script = """<script>
        const API_URL = "YOUR_API_URL_HERE";
        let currentGeneration = null;
        let globalLibrary = [];

        async function init() {
            try {
                let res = await fetch('/api/library');
                globalLibrary = await res.json();
                renderLibrary();
            } catch (e) { console.error("Could not load library:", e); }
        }

        async function saveLibrary() {
            await fetch('/api/library', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(globalLibrary)
            });
        }

        function switchTab(tab) {
            document.getElementById('nav-generate').classList.remove('active');
            document.getElementById('nav-library').classList.remove('active');
            document.getElementById('nav-favorites').classList.remove('active');
            document.getElementById('nav-' + tab).classList.add('active');
            
            document.getElementById('tab-generate').style.display = 'none';
            document.getElementById('tab-library').style.display = 'none';
            document.getElementById('tab-favorites').style.display = 'none';
            document.getElementById('tab-' + tab).style.display = 'block';
            
            if(tab === 'library' || tab === 'favorites') { renderLibrary(); }
        }

        function setPrompt(text) { document.getElementById('prompt').value = text; }

        async function toggleFavorite(id) {
            const item = globalLibrary.find(i => i.id === id);
            if(item) {
                item.fav = !item.fav;
                renderLibrary();
                await saveLibrary();
            }
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

        async function blobToBase64(blobUrl) {
            let blob = await fetch(blobUrl).then(r => r.blob());
            return new Promise((resolve) => {
                let reader = new FileReader();
                reader.onloadend = () => resolve(reader.result);
                reader.readAsDataURL(blob);
            });
        }

        async function addCurrentToLibrary() {
            if(!currentGeneration) return;
            const btn = document.getElementById('saveLibraryBtn');
            btn.innerText = "Saving to disk...";
            btn.disabled = true;

            const b64_base = await blobToBase64(currentGeneration.base);
            const b64_lora = await blobToBase64(currentGeneration.lora);

            const res = await fetch('/api/download', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ base64_base: b64_base, base64_lora: b64_lora })
            });
            const paths = await res.json();

            globalLibrary.unshift({
                id: Date.now(),
                title: "Live Generation",
                prompt: currentGeneration.prompt,
                base: paths.base_url,
                lora: paths.lora_url,
                fav: false
            });
            
            await saveLibrary();
            btn.innerText = "✓ Permanently Saved!";
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
                
                currentGeneration = { prompt: prompt, base: baseUrl, lora: loraUrl };
                saveBtn.style.display = "inline-block";
                saveBtn.innerText = "📥 Add to Library (Permanently)";
                saveBtn.disabled = false;
            } catch (e) {
                alert("Failed: " + e.message);
                loaderBase.style.display = "none"; loaderLora.style.display = "none";
            } finally {
                btn.disabled = false; btn.innerText = "✨ Generate (2)";
            }
        }

        init();
    </script>"""

new_html = html[:script_start] + new_script + html[script_end:]
with open("../frontend/index.html", "w") as f:
    f.write(new_html)
