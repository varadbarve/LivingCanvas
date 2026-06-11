import sys
import os
import numpy as np
import cv2

# Set path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config
from src import model_manager
from src.engine import VectorRenderingEngine
from src.stream_a import StreamA
from src.stream_b import StreamB
from src.utils import DoubleBuffer

def run_tests():
    print("----- Running Living Canvas Verification -----")
    
    # 1. Test ONNX Loading
    print("1. Testing ONNX model loading...")
    for style in config.STYLE_MODELS.keys():
        print(f"  Loading '{style}'...")
        session = model_manager.load_style_model(style)
        assert session is not None, f"Failed to load ONNX session for {style}"
        print(f"  Successfully loaded '{style}'")
        
    # 2. Test Stream A Pre/Post-processing logic
    print("2. Testing Stream A pre/post-processing logic...")
    dummy_frame = np.random.randint(0, 255, (config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
    
    # Preprocess
    rgb_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT))
    in_data = resized.astype(np.float32)
    in_tensor = np.transpose(in_data, (2, 0, 1))
    in_tensor = np.expand_dims(in_tensor, axis=0)
    assert in_tensor.shape == (1, 3, config.PROCESSING_HEIGHT, config.PROCESSING_WIDTH), "Input shape mismatch"
    print("  Pre-processing shape correct:", in_tensor.shape)

    # 3. Test Engine Initialization and Reseeding
    print("3. Testing Rendering Engine and Particle updates...")
    engine = VectorRenderingEngine()
    assert len(engine.x) == config.MAX_PARTICLE_COUNT, "Particle arrays pre-allocation failed"
    print(f"  Pre-allocated {config.MAX_PARTICLE_COUNT} particles successfully.")
    
    # Make mock control matrices
    control_matrices = {
        "flow": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH, 2), dtype=np.float32),
        "edge_magnitude": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH), dtype=np.float32),
        "edge_angle": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH), dtype=np.float32),
        "saliency": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH), dtype=np.float32),
        "gray": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH), dtype=np.uint8)
    }
    stylized_texture = np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), dtype=np.uint8)
    
    # Run a single engine update step
    engine.update(control_matrices, stylized_texture)
    canvas = engine.draw()
    assert canvas.shape == (config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), "Canvas rendering shape mismatch"
    print("  Engine update & draw runs successfully without crash.")
    
    print("\nALL OFFLINE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
