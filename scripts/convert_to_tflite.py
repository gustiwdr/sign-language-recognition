#!/usr/bin/env python3
# filepath: /root/projects/events/apac-google/sign-language-recognition/scripts/convert_to_tflite.py
"""
Script to convert a Keras model to TensorFlow Lite format.

This script loads a trained model and converts it to TFLite format for mobile deployment,
with options for quantization and optimization.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert Keras model to TFLite format')
    parser.add_argument('--model', type=str, default='../models/final_model.h5',
                        help='Path to the trained Keras model')
    parser.add_argument('--output', type=str, default='../models/final_model.tflite',
                        help='Path to save the TFLite model')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply quantization to reduce model size')
    parser.add_argument('--optimize', action='store_true',
                        help='Apply optimizations for inference')
    parser.add_argument('--metadata', action='store_true',
                        help='Add metadata to the model')
    return parser.parse_args()

def convert_to_tflite(model_path, output_path, quantize=False, optimize=False, add_metadata=False):
    """
    Convert a Keras model to TFLite format.
    """
    try:
        # Load the model
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully.")
        
        # Create a converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        converter._experimental_lower_tensor_list_ops = False
        
        # Apply optimizations if requested
        if optimize:
            print("Applying inference optimizations...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Apply quantization if requested
        if quantize:
            print("Applying quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # CATATAN: Jika menggunakan quantization, tetap pertahankan SELECT_TF_OPS
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.representative_dataset = representative_dataset_gen
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
        # Convert the model
        print("Converting model to TFLite format...")
        tflite_model = converter.convert()
        
        # Add metadata if requested
        if add_metadata:
            print("Adding metadata to model...")
            # This would require additional code to add appropriate metadata
            # For example, input/output tensor details, model description, etc.
            # You would use the TFLite Metadata Writer API
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Calculate and print model size
        model_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"Model successfully converted and saved to {output_path}")
        print(f"Model size: {model_size:.2f} MB")
        
        return True
    
    except Exception as e:
        print(f"Error converting model: {e}")
        return False

def representative_dataset_gen():
    """
    Generate representative dataset for quantization.
    
    This function should yield calibration data that represents the expected input.
    """
    # This is a placeholder - you would need to implement this properly
    # with actual representative data from your dataset
    for _ in range(100):
        # Generate random data in the expected input shape of your model
        # Adjust shape according to your model's input shape
        data = np.random.rand(1, 100, 99)  # Example shape: (batch, frames, features)
        yield [data.astype(np.float32)]

def main():
    """Main function."""
    args = parse_arguments()
    
    print("Starting model conversion to TFLite...")
    success = convert_to_tflite(
        args.model, 
        args.output, 
        quantize=args.quantize, 
        optimize=args.optimize,
        add_metadata=args.metadata
    )
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main()