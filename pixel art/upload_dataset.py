import modal
import os

app = modal.App("upload-dataset-script")
vol = modal.Volume.from_name("flux-lora-volume")

@app.local_entrypoint()
def main():
    print("Uploading dataset to volume...")
    with vol.batch_upload() as batch:
        for root, dirs, files in os.walk("dataset"):
            for file in files:
                local_path = os.path.join(root, file)
                remote_path = f"/dataset/pixel_art/{local_path}"
                batch.put_file(local_path, remote_path)
    print("Done!")
