import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Resolutions
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Processing Resolution (Stream A / B inputs)
PROCESSING_WIDTH = 320
PROCESSING_HEIGHT = 240

# High-resolution rendering canvas
RENDER_WIDTH = 800
RENDER_HEIGHT = 600

# ONNX Style Models
STYLE_MODELS = {
    "Starry Night": "rain-princess-9.onnx",
    "The Scream": "FNS-The-Scream.onnx",
    "Modernist Geometric": "candy-9.onnx"
}

STYLE_MODEL_URLS = {
    "rain-princess-9.onnx": "https://huggingface.co/onnxmodelzoo/rain-princess-9/resolve/main/rain-princess-9.onnx",
    "FNS-The-Scream.onnx": "https://raw.githubusercontent.com/ChangweiZhang/Awesome-ONNX-Models/master/FNS-The-Scream.onnx",
    "candy-9.onnx": "https://huggingface.co/onnxmodelzoo/candy-9/resolve/main/candy-9.onnx"
}

# Dense Optical Flow (Farneback parameters)
FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2
FLOW_FLAGS = 0

# Saliency Analysis
SALIENCY_THRESHOLD = 0.5  # Foreground threshold for MediaPipe Selfie Segmentation

# Vector Rendering Engine Defaults
DEFAULT_PARTICLE_COUNT = 1500
MAX_PARTICLE_COUNT = 4000
MIN_PARTICLE_COUNT = 500

DEFAULT_ALPHA = 1.5       # Paint viscosity/advection multiplier
MIN_ALPHA = 0.1
MAX_ALPHA = 5.0

PARTICLE_MIN_TTL = 30     # Minimum lifetime in frames
PARTICLE_MAX_TTL = 100    # Maximum lifetime in frames

# Brush stroke styling parameters
BRUSH_COARSE_SIZE = 14
BRUSH_FINE_SIZE = 5

BRUSH_COARSE_LENGTH = 25
BRUSH_FINE_LENGTH = 8

# Edge alignment influence [0.0 to 1.0]
DEFAULT_EDGE_INFLUENCE = 0.8
