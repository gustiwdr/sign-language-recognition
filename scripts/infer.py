#!/usr/bin/env python3
"""
Inference script for BISINDO Sign Language Recognition model.

This script uses a trained model to make predictions on new sign language data.
It can process either individual files or entire directories of landmark data.

Usage:
    python infer.py --model /path/to/model --input /path/to/input/data --output /path/to/results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('infer')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inference for BISINDO Sign Language Recognition')
    parser.add_argument('--model', type=str, default='../models',
                        help='Path to the directory containing the trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input file or directory with landmark data')
    parser.add_argument('--output', type=str, default='../predictions',
                        help='Path to save prediction results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate and save visualizations of predictions')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Show top K predictions for each sample')
    return parser.parse_args()

def preprocess_landmarks(landmark_file, max_frames=100, num_landmarks=33, num_dimensions=3):
    """
    Preprocess landmark data from a single file for inference.
    
    Args:
        landmark_file: Path to the landmark file (.npy)
        max_frames: Maximum number of frames to keep
        num_landmarks: Number of landmarks in each frame
        num_dimensions: Number of dimensions for each landmark
        
    Returns:
        Preprocessed landmark data ready for model input
    """
    try:
        # Load landmarks data
        landmarks = np.load(landmark_file)
        
        # Handle different possible data shapes
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)
        
        # Ensure landmarks have consistent dimensions
        if landmarks.shape[1] != num_landmarks * num_dimensions:
            if landmarks.shape[1] % num_dimensions == 0:
                new_num_landmarks = landmarks.shape[1] // num_dimensions
                landmarks = landmarks.reshape(landmarks.shape[0], new_num_landmarks, num_dimensions)
            else:
                logger.error(f"Incompatible landmark shape: {landmarks.shape}")
                return None
        else:
            landmarks = landmarks.reshape(landmarks.shape[0], num_landmarks, num_dimensions)
        
        # Normalize coordinates to [0, 1] range
        for i in range(num_dimensions):
            min_val = np.min(landmarks[:, :, i])
            max_val = np.max(landmarks[:, :, i])
            if max_val > min_val:
                landmarks[:, :, i] = (landmarks[:, :, i] - min_val) / (max_val - min_val)
        
        # Handle sequence length (padding or truncating)
        if landmarks.shape[0] > max_frames:
            landmarks = landmarks[:max_frames]
        elif landmarks.shape[0] < max_frames:
            padding = np.zeros((max_frames - landmarks.shape[0], landmarks.shape[1], landmarks.shape[2]))
            landmarks = np.vstack([landmarks, padding])
        
        # Reshape to the format expected by the model
        landmarks = landmarks.reshape(1, max_frames, -1)  # Add batch dimension and flatten landmarks
        
        return landmarks
        
    except Exception as e:
        logger.error(f"Error processing {landmark_file}: {e}")
        return None

def visualize_prediction(landmark_file, prediction, class_names, output_path):
    """
    Create and save a visualization of the prediction alongside the landmark data.
    
    Args:
        landmark_file: Path to the landmark file
        prediction: Dictionary containing prediction results
        class_names: List of class names
        output_path: Directory to save the visualization
    """
    try:
        # Load landmark data
        landmarks = np.load(landmark_file)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Section 1: Display landmark visualization
        plt.subplot(2, 1, 1)
        
        # Display a representative frame from the landmarks
        if landmarks.ndim > 1:
            frame_idx = min(10, landmarks.shape[0] - 1)  # Get a frame that's not at the beginning
            frame_data = landmarks[frame_idx]
        else:
            frame_data = landmarks
        
        # Create a blank image
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Process points for visualization
        if frame_data.ndim > 1:
            points = frame_data[:, :2]  # Assuming first two columns are x,y
        else:
            # If flat array, try to reshape assuming groups of 3 (x,y,z)
            points = frame_data.reshape(-1, 3)[:, :2]
        
        # Normalize to fit in our image
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        range_vals = max_vals - min_vals
        normalized_points = (points - min_vals) / range_vals if np.any(range_vals) else points
        
        # Plot points
        for pt in normalized_points:
            x, y = int(pt[0] * 350 + 25), int(pt[1] * 350 + 25)
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        
        plt.imshow(img)
        plt.title('Landmark Visualization')
        plt.axis('off')
        
        # Section 2: Display prediction results as a bar chart
        plt.subplot(2, 1, 2)
        
        top_classes = [pred['class'] for pred in prediction['top_predictions']]
        top_probs = [pred['probability'] for pred in prediction['top_predictions']]
        
        y_pos = np.arange(len(top_classes))
        plt.barh(y_pos, top_probs, align='center')
        plt.yticks(y_pos, top_classes)
        plt.xlabel('Probability')
        plt.title('Top Predictions')
        
        # Add text annotation with the predicted class
        plt.figtext(0.5, 0.01, f"Predicted: {prediction['top_predictions'][0]['class']} "
                               f"(Confidence: {prediction['top_predictions'][0]['probability']:.2f})",
                    ha='center', fontsize=12, bbox={'facecolor': 'yellow', 'alpha': 0.5})
        
        plt.tight_layout()
        
        # Save the visualization
        file_name = os.path.basename(landmark_file).replace('.npy', '_prediction.png')
        vis_path = os.path.join(output_path, file_name)
        plt.savefig(vis_path)
        plt.close()
        
        logger.info(f"Visualization saved to {vis_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")

def run_inference(model, class_names, input_path, output_path, top_k=3, visualize=False):
    """
    Run inference on input data.
    
    Args:
        model: Loaded TensorFlow model
        class_names: List of class names
        input_path: Path to input file or directory
        output_path: Path to save results
        top_k: Number of top predictions to include
        visualize: Whether to generate visualizations
        
    Returns:
        List of prediction results
    """
    results = []
    
    # Check if input is a directory or a file
    if os.path.isdir(input_path):
        input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                       if f.endswith('.npy') and not f.endswith(':Zone.Identifier')]
    else:
        input_files = [input_path]
    
    logger.info(f"Processing {len(input_files)} input files...")
    
    for file_path in tqdm(input_files, desc="Running inference"):
        file_name = os.path.basename(file_path)
        
        # Preprocess the landmark data
        processed_data = preprocess_landmarks(file_path)
        if processed_data is None:
            logger.warning(f"Skipping {file_name} due to preprocessing error")
            continue
        
        # Make prediction
        prediction_prob = model.predict(processed_data, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = prediction_prob.argsort()[-top_k:][::-1]
        top_predictions = [
            {
                "class": class_names[idx],
                "probability": float(prediction_prob[idx])
            }
            for idx in top_indices
        ]
        
        # Create result dictionary
        result = {
            "file_name": file_name,
            "input_path": file_path,
            "top_predictions": top_predictions,
            "predicted_class": class_names[top_indices[0]],
            "confidence": float(prediction_prob[top_indices[0]]),
            "timestamp": datetime.now().isoformat()
        }
        
        results.append(result)
        
        # Generate visualization if requested
        if visualize:
            visualize_prediction(file_path, result, class_names, output_path)
        
        # Print prediction
        logger.info(f"Prediction for {file_name}: {result['predicted_class']} "
                   f"(Confidence: {result['confidence']:.4f})")
    
    return results

def main(args):
    """Main function for inference."""
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Results will be saved to {args.output}")
    
    # Load model
    try:
        model_path = os.path.join(args.model, 'best_model.h5')
        if os.path.exists(model_path):
            model = load_model(model_path)
            logger.info("Loaded best model from training")
        else:
            model_path = os.path.join(args.model, 'final_model.h5')
            model = load_model(model_path)
            logger.info("Loaded final model")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Make sure the model file exists and is valid")
        sys.exit(1)
    
    # Load class names
    try:
        class_names_path = os.path.join(args.model, 'classes.npy')
        class_names = np.load(class_names_path, allow_pickle=True)
        logger.info(f"Loaded {len(class_names)} classes")
    except Exception as e:
        logger.error(f"Error loading class names: {e}")
        logger.error("Make sure the classes.npy file exists in the model directory")
        sys.exit(1)
    
    # Check if input path exists
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Run inference
    results = run_inference(
        model=model,
        class_names=class_names,
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        visualize=args.visualize
    )
    
    # Save results to JSON file
    results_file = os.path.join(args.output, 'inference_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # Generate summary
    total_samples = len(results)
    if total_samples > 0:
        # Count predictions per class
        class_counts = {}
        for result in results:
            predicted_class = result['predicted_class']
            if predicted_class in class_counts:
                class_counts[predicted_class] += 1
            else:
                class_counts[predicted_class] = 1
        
        # Calculate average confidence
        avg_confidence = sum(result['confidence'] for result in results) / total_samples
        
        # Save summary
        summary = {
            "total_samples": total_samples,
            "class_distribution": class_counts,
            "average_confidence": avg_confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = os.path.join(args.output, 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_file}")
        
        # Log summary
        logger.info("\n--- Inference Summary ---")
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Average confidence: {avg_confidence:.4f}")
        logger.info("Class distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {cls}: {count} ({count/total_samples*100:.1f}%)")
    
    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)