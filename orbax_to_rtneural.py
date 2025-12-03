#!/usr/bin/env python3
"""
Convert Orbax/Flax checkpoint to RTNeural JSON format.

Usage:
    # From Orbax checkpoint directory:
    python orbax_to_rtneural.py checkpoint_dir output.json
    
    # From Flax JSON export:
    python orbax_to_rtneural.py flax_policy.json output.json
"""

import json
import numpy as np
from pathlib import Path
import sys


def load_orbax_checkpoint(checkpoint_path: Path) -> dict:
    """Load weights from Orbax checkpoint directory."""
    try:
        import orbax.checkpoint as ocp
        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(str(checkpoint_path))
    except ImportError:
        print("Orbax not installed. Install with: pip install orbax-checkpoint")
        sys.exit(1)


def load_flax_json(json_path: Path) -> dict:
    """Load weights from Flax JSON export."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_layer_params(params: dict) -> list:
    """
    Extract layer parameters from Flax params structure.
    
    Expects structure like:
    {
        'params': {
            'hidden_0': {'kernel': [...], 'bias': [...]},
            'hidden_1': {...},
            ...
        }
    }
    or directly:
    {
        'hidden_0': {'kernel': [...], 'bias': [...]},
        ...
    }
    """
    # Navigate to the actual layer dict
    if 'params' in params and 'params' in params['params']:
        layer_dict = params['params']['params']
    elif 'params' in params:
        layer_dict = params['params']
    else:
        layer_dict = params
    
    # Find all hidden layers
    layers = []
    i = 0
    while True:
        layer_name = f'hidden_{i}'
        if layer_name not in layer_dict:
            # Also try just numbered layers
            if str(i) not in layer_dict and f'Dense_{i}' not in layer_dict:
                break
            layer_name = str(i) if str(i) in layer_dict else f'Dense_{i}'
        
        layer_params = layer_dict[layer_name]
        kernel = layer_params['kernel']
        bias = layer_params['bias']
        
        # Convert to lists if numpy arrays
        if hasattr(kernel, 'tolist'):
            kernel = kernel.tolist()
        if hasattr(bias, 'tolist'):
            bias = bias.tolist()
            
        layers.append({
            'kernel': kernel,
            'bias': bias
        })
        i += 1
    
    return layers


def convert_to_rtneural(
    source,
    observation_size: int = 720,
    observation_history: int = 20,
    kp: float = 7.5,
    kd: float = 0.25,
    action_scale: float = 0.5,
    activation: str = "elu",
    default_joint_pos: list = None,
    joint_upper_limits: list = None,
    joint_lower_limits: list = None,
) -> dict:
    """
    Convert Flax/Orbax checkpoint to RTNeural format.
    
    Args:
        source: Loaded checkpoint (can be list or dict)
        observation_size: Total observation vector size (default 720 = 20 frames × 36 features)
        observation_history: Number of stacked frames
        kp: Position gain
        kd: Derivative gain
        action_scale: Action scaling factor
        activation: Activation function for hidden layers
        default_joint_pos: Default joint positions (12 values)
        joint_upper_limits: Joint upper limits (12 values)
        joint_lower_limits: Joint lower limits (12 values)
    """
    
    # Default joint values for Pupper
    if default_joint_pos is None:
        default_joint_pos = [0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 
                            0.26, 0.0, -0.52, -0.26, 0.0, 0.52]
    if joint_upper_limits is None:
        joint_upper_limits = [0.8, 1.2, 0.0, 0.0, 1.2, 1.2, 
                             0.8, 1.2, 0.0, 0.0, 1.2, 1.2]
    if joint_lower_limits is None:
        joint_lower_limits = [-0.0, -1.2, -1.2, -0.8, -1.2, -0.0, 
                             -0.0, -1.2, -1.2, -0.8, -1.2, -0.0]
    
    # Handle Orbax checkpoint structure: [normalizer_dict, model_dict]
    if isinstance(source, list):
        if len(source) >= 2:
            normalizer_data = source[0]  # {count, mean, std, summed_variance}
            model_data = source[1]       # {policy, value}
        else:
            raise ValueError(f"Unexpected list structure with {len(source)} elements")
    else:
        # Fallback for dict structure
        normalizer_data = source.get('normalizer', {})
        model_data = source
    
    # Extract policy params
    if isinstance(model_data, dict) and 'policy' in model_data:
        policy_params = model_data['policy']
    else:
        policy_params = model_data
    
    # Extract normalizer (handle numpy arrays)
    normalizer = {}
    if isinstance(normalizer_data, dict):
        if 'mean' in normalizer_data:
            mean = normalizer_data['mean']
            normalizer['mean'] = mean.tolist() if hasattr(mean, 'tolist') else mean
        if 'std' in normalizer_data:
            std = normalizer_data['std']
            normalizer['std'] = std.tolist() if hasattr(std, 'tolist') else std
    # Extract layers
    layers = extract_layer_params(policy_params)
    
    if not layers:
        raise ValueError("No layers found in checkpoint. Check the structure.")
    
    # Build RTNeural format
    rtneural = {
        "in_shape": [1, observation_size],
        "observation_history": observation_history,
        "kp": kp,
        "kd": kd,
        "action_scale": action_scale,
        "use_imu": True,
        "default_joint_pos": default_joint_pos,
        "joint_upper_limits": joint_upper_limits,
        "joint_lower_limits": joint_lower_limits,
        "normalizer": normalizer,
        "layers": []
    }
    
    for i, layer in enumerate(layers):
        kernel = layer['kernel']
        bias = layer['bias']
        
        # Determine dimensions
        in_features = len(kernel)
        out_features = len(kernel[0]) if kernel else len(bias)
        
        # Last layer has no activation
        is_last = (i == len(layers) - 1)
        layer_activation = "" if is_last else activation
        
        rtneural_layer = {
            "type": "dense",
            "shape": [in_features, out_features],
            "activation": layer_activation,
            "weights": [kernel, bias]
        }
        
        rtneural["layers"].append(rtneural_layer)
    
    return rtneural


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nExample:")
        print("  python orbax_to_rtneural.py my_checkpoint/ pure_rl_policy.json")
        print("  python orbax_to_rtneural.py flax_export.json pure_rl_policy.json")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    # Load source
    if input_path.is_dir():
        print(f"Loading Orbax checkpoint from: {input_path}")
        source = load_orbax_checkpoint(input_path)
    else:
        print(f"Loading JSON from: {input_path}")
        source = load_flax_json(input_path)
    
    # Convert
    print("Converting to RTNeural format...")
    rtneural = convert_to_rtneural(source)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(rtneural, f)
    
    print(f"\nSaved to: {output_path}")
    print(f"  in_shape: {rtneural['in_shape']}")
    print(f"  observation_history: {rtneural['observation_history']}")
    print(f"  Number of layers: {len(rtneural['layers'])}")
    for i, layer in enumerate(rtneural['layers']):
        print(f"    Layer {i}: {layer['type']} {layer['shape']} activation='{layer['activation']}'")
    if rtneural.get('normalizer', {}).get('mean'):
        print(f"  Normalizer: {len(rtneural['normalizer']['mean'])} values")
    
    print("\n✓ Conversion complete!")


if __name__ == "__main__":
    main()

