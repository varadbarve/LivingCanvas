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
        
    # 2. Test Stream A Pre/Post-processing logic and Inference
    print("2. Testing Stream A pre/post-processing logic and ONNX inference...")
    dummy_frame = np.random.randint(0, 255, (config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
    
    # Preprocess
    session = model_manager.load_style_model("Starry Night")
    input_shape = session.get_inputs()[0].shape
    model_h = input_shape[2] if (len(input_shape) > 2 and isinstance(input_shape[2], int)) else 224
    model_w = input_shape[3] if (len(input_shape) > 3 and isinstance(input_shape[3], int)) else 224
    
    rgb_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (model_w, model_h))
    in_data = resized.astype(np.float32)
    in_tensor = np.transpose(in_data, (2, 0, 1))
    in_tensor = np.expand_dims(in_tensor, axis=0)
    assert in_tensor.shape == (1, 3, model_h, model_w), "Input shape mismatch"
    print("  Pre-processing shape correct:", in_tensor.shape)
    # Run test inference
    model_inputs = {session.get_inputs()[0].name: in_tensor}
    model_outputs = session.run(None, model_inputs)
    assert len(model_outputs) > 0, "No outputs from ONNX run"
    print("  ONNX model run test passed successfully.")
    
    # Test Custom CV Styles on StreamA
    from src.utils import DoubleBuffer
    from src.stream_a import StreamA
    dummy_in = DoubleBuffer()
    dummy_out = DoubleBuffer()
    stream_a_test = StreamA(dummy_in, dummy_out)
    
    for style in ["Cartoon", "Oil Pastel Painting", "ASCII Character Vision", "Line Drawing"]:
        print(f"  Testing CV style processing for '{style}'...")
        tex = stream_a_test.apply_cv_style(dummy_frame, style)
        assert tex.shape == (config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), f"CV style output shape mismatch for {style}"
        print(f"    Passed '{style}'")

    # 3. Test Engine Initialization, Reseeding, and Style drawing
    print("3. Testing Rendering Engine and Particle updates for all styles...")
    engine = VectorRenderingEngine()
    assert len(engine.x) == config.MAX_PARTICLE_COUNT, "Particle arrays pre-allocation failed"
    print(f"  Pre-allocated {config.MAX_PARTICLE_COUNT} particles successfully.")
    
    # Make mock control matrices
    control_matrices = {
        "flow": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH, 2), dtype=np.float32),
        "edge_magnitude": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH), dtype=np.float32),
        "edge_angle": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH), dtype=np.float32),
        "saliency": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH), dtype=np.float32),
        "gray": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH), dtype=np.uint8),
        "color": np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), dtype=np.uint8)
    }
    stylized_texture = np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), dtype=np.uint8)
    
    # Run a single engine update step for every style
    for style in config.ALL_STYLES:
        print(f"  Testing update & draw for style '{style}'...")
        engine.update(control_matrices, stylized_texture, style_name=style)
        canvas = engine.draw()
        assert canvas.shape == (config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), f"Canvas rendering shape mismatch for {style}"
        print(f"    Passed '{style}'")
    
    print("\nALL OFFLINE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
