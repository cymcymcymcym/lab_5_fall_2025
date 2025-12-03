#!/usr/bin/env python3
"""
Convert pure RL policy from Flax/JAX format to RTNeural format.
"""

import json
import numpy as np


def convert_policy(input_path: str, output_path: str):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Extract the network params
    params = data['policy']['params']['params']
    normalizer = data.get('normalizer', {})
    
    # Build RTNeural format
    rtneural = {
        # Model metadata
        "in_shape": [1, 720],  # 20 frames * 36 features
        "observation_history": 20,
        
        # Controller parameters - adjust these as needed
        "kp": 7.5,
        "kd": 0.25,
        "action_scale": 0.5,
        "use_imu": True,
        
        # Default joint positions (typical pupper values)
        "default_joint_pos": [0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52],
        
        # Joint limits
        "joint_upper_limits": [0.8, 1.2, 0.0, 0.0, 1.2, 1.2, 0.8, 1.2, 0.0, 0.0, 1.2, 1.2],
        "joint_lower_limits": [-0.0, -1.2, -1.2, -0.8, -1.2, -0.0, -0.0, -1.2, -1.2, -0.8, -1.2, -0.0],
        
        # Normalizer (pass through from original)
        "normalizer": normalizer,
        
        # RTNeural layers
        "layers": []
    }
    
    # Layer names in order
    layer_names = ['hidden_0', 'hidden_1', 'hidden_2', 'hidden_3', 'hidden_4']
    
    for i, layer_name in enumerate(layer_names):
        layer_params = params[layer_name]
        kernel = layer_params['kernel']  # shape: [in_features, out_features]
        bias = layer_params['bias']      # shape: [out_features]
        
        # RTNeural expects weights in column-major order for Dense layers
        # The kernel is [in, out], RTNeural wants weights as list of lists
        kernel_np = np.array(kernel)
        
        # Determine activation (last layer has no activation or tanh)
        is_last = (i == len(layer_names) - 1)
        activation = "" if is_last else "elu"  # Using ELU like typical RL policies
        
        layer = {
            "type": "dense",
            "shape": list(kernel_np.shape),
            "weights": kernel,  # Keep as nested list
            "bias": bias,
        }
        
        rtneural["layers"].append(layer)
        
        # Add activation layer after each dense (except last)
        if not is_last:
            rtneural["layers"].append({
                "type": "activation",
                "activation": activation,
                "shape": [kernel_np.shape[1]]
            })
    
    # Write output
    with open(output_path, 'w') as f:
        json.dump(rtneural, f)
    
    print(f"Converted {input_path} -> {output_path}")
    print(f"Input shape: {rtneural['in_shape']}")
    print(f"Number of layers: {len(rtneural['layers'])}")
    print(f"Normalizer mean length: {len(normalizer.get('mean', []))}")
    print(f"Normalizer std length: {len(normalizer.get('std', []))}")


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "pure_rl_policy.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "pure_rl_policy_rtneural.json"
    convert_policy(input_file, output_file)

