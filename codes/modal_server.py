import modal
import os
import io
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = modal.App("gojo-lora-server")

# 1. The GPU Image Definition
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "peft",
        "fastapi"
    )
)

volume = modal.Volume.from_name("flux-lora-volume")
API_TOKEN = "YOUR_SECRET_TOKEN_HERE"

# 2. The GPU Worker Class
@app.cls(
    image=gpu_image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1200,         
    max_containers=3,     
    min_containers=0      
)
class GojoGenerator:
    @modal.enter()
    def load_model(self):
        print("Initializing FLUX.1-dev and Gojo LoRA...")
        import torch
        from diffusers import FluxPipeline
        
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")
        self.pipe.load_lora_weights("/data/output/gojo_lora/gojo_lora.safetensors", adapter_name="gojo")
        print("Model loaded and ready!")

    @modal.method()
    def generate(self, prompt: str, seed: int, use_lora: bool):
        import torch
        generator = torch.Generator("cuda").manual_seed(seed)
        lora_scale = 1.0 if use_lora else 0.0
        
        img = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            joint_attention_kwargs={"scale": lora_scale},
            generator=generator
        ).images[0]
        
        byte_stream = io.BytesIO()
        img.save(byte_stream, format="PNG")
        return byte_stream.getvalue()

# 3. The Lightweight Web API (Handles CORS & Security)
web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Fixes the localhost CORS NetworkError!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.post("/")
async def api_endpoint(request: Request):
    # Security Check
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header != f"Bearer {API_TOKEN}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing API Token"
        )
        
    # Parse Request
    body = await request.json()
    user_prompt = body.get("prompt", "")
    seed = body.get("seed", 42)
    use_lora = body.get("use_lora", True)
    
    # SYSTEM PROMPT INJECTION
    # This automatically gives the model his core looks so the user just types the action!
    system_prefix = "Satoru Gojo, adult male, spiky pure-white hair, icy-blue eyes, wearing dark high-collared Jujutsu Kaisen uniform, masterpiece anime illustration, 8k resolution, highly detailed. "
    
    final_prompt = system_prefix + user_prompt
    
    print(f"Authorized. Spawning GPU worker. Final Prompt: {final_prompt}, LoRA: {use_lora}")
    
    # Call the massive GPU worker asynchronously so we don't crash the CPU container!
    image_bytes = await GojoGenerator().generate.remote.aio(final_prompt, seed, use_lora)
    
    return Response(content=image_bytes, media_type="image/png")

web_image = modal.Image.debian_slim(python_version="3.10").pip_install("fastapi")

@app.function(image=web_image, timeout=1200)
@modal.asgi_app()
def serve():
    return web_app
