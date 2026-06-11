import os
import urllib.request
from src import config

HAS_ORT = False
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None

def get_model_path(style_name):
    """Get the local file path for a style model, downloading it if not present."""
    if style_name not in config.STYLE_MODELS:
        raise ValueError(f"Unknown style: {style_name}")
    
    filename = config.STYLE_MODELS[style_name]
    local_path = os.path.join(config.MODELS_DIR, filename)
    
    if not os.path.exists(config.MODELS_DIR):
        os.makedirs(config.MODELS_DIR)
        
    if not os.path.exists(local_path):
        url = config.STYLE_MODEL_URLS[filename]
        print(f"Downloading pre-trained style model '{style_name}' from {url}...")
        print("This may take a moment (~6.7 MB)...")
        
        # User-agent header to avoid blocks from hosting providers
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                total_size = int(response.info().get('Content-Length', 0))
                bytes_downloaded = 0
                block_size = 1024 * 64
                
                with open(local_path, 'wb') as out_file:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        bytes_downloaded += len(buffer)
                        out_file.write(buffer)
                        
                        if total_size > 0:
                            percent = (bytes_downloaded / total_size) * 100
                            print(f"\rDownloading: {percent:.1f}% ({bytes_downloaded // 1024} KB / {total_size // 1024} KB)", end="")
            print(f"\nSuccessfully downloaded '{style_name}' to {local_path}")
        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError(f"Failed to download style model '{style_name}': {e}")
            
    return local_path

def load_style_model(style_name):
    """Load and return the ONNX InferenceSession for the given style."""
    if not HAS_ORT or ort is None:
        raise RuntimeError("ONNX Runtime is not available on this system.")
        
    model_path = get_model_path(style_name)
    
    # We explicitly specify CPU Execution Provider for consistency on consumer laptops
    providers = ['CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        return session
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ONNX session for '{style_name}': {e}")

if __name__ == "__main__":
    # Test model downloader and loader
    for style in config.STYLE_MODELS.keys():
        print(f"Verifying {style}...")
        get_model_path(style)
