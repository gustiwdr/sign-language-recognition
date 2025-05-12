#!/usr/bin/env python3
"""
Preprocessing script for BISINDO Sign Language Recognition dataset.

This script processes raw BISINDO sign language data, extracts landmarks,
and prepares the dataset for training. It handles the conversion of raw data
to a structured format with metadata.

Usage:
    python preprocess.py --input /path/to/raw/data --output /path/to/processed/data
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import argparse
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('preprocess')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess BISINDO Sign Language data')
    parser.add_argument('--input', type=str, default='../data/raw/BISINDO',
                        help='Path to the raw data directory')
    parser.add_argument('--output', type=str, default='../data/processed',
                        help='Path to save processed data')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of sample landmarks')
    return parser.parse_args()

def create_directories(processed_path, landmarks_path):
    """Create necessary directories if they don't exist."""
    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(landmarks_path, exist_ok=True)
    logger.info(f"Created directories: {processed_path}, {landmarks_path}")

def list_classes(base_path):
    """List all classes (signs) in the dataset."""
    classes = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    logger.info(f"Found {len(classes)} classes: {classes}")
    return classes

def process_landmarks(base_path, landmarks_path, classes):
    """
    Process landmark data for all classes.
    
    Args:
        base_path: Path to the raw data
        landmarks_path: Path to save processed landmarks
        classes: List of class names
        
    Returns:
        List of dictionaries containing metadata for each processed sample
    """
    data = []
    for class_name in tqdm(classes, desc="Processing classes"):
        class_path = os.path.join(base_path, class_name)
        
        # Process each subfolder (usually representing different people/samples)
        for person_id in os.listdir(class_path):
            person_path = os.path.join(class_path, person_id)
            
            if os.path.isdir(person_path):
                # Process each NPY file in the subfolder
                for file_name in os.listdir(person_path):
                    if file_name.endswith('.npy') and not file_name.endswith(':Zone.Identifier'):
                        file_path = os.path.join(person_path, file_name)
                        
                        try:
                            # Load the landmark data
                            landmarks = np.load(file_path)
                            
                            # Create a unique identifier for this sample
                            sample_id = f"{class_name}_{person_id}_{file_name.replace('.npy', '')}"
                            
                            # Save processed landmarks to the landmarks directory
                            output_file = os.path.join(landmarks_path, f"{sample_id}.npy")
                            np.save(output_file, landmarks)
                            
                            # Add metadata to our list
                            data.append({
                                'sample_id': sample_id,
                                'class': class_name,
                                'person_id': person_id,
                                'original_file': file_name,
                                'n_frames': landmarks.shape[0] if landmarks.ndim > 1 else 1,
                                'landmark_path': output_file
                            })
                            
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Total processed samples: {len(data)}")
    return data

def analyze_dataset(df):
    """
    Analyze the dataset and print statistics.
    
    Args:
        df: DataFrame containing metadata
    """
    logger.info("\nSample distribution per class:")
    class_counts = df['class'].value_counts().sort_index()
    logger.info(class_counts)
    
    logger.info("\nBasic statistics for number of frames per sample:")
    logger.info(df['n_frames'].describe())

def visualize_class_distribution(df, output_path):
    """
    Visualize the class distribution in the dataset.
    
    Args:
        df: DataFrame containing metadata
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(14, 6))
    class_counts = df['class'].value_counts().sort_index()
    class_counts.plot(kind='bar')
    plt.title('Number of Samples per Sign Class')
    plt.xlabel('Sign Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    dist_file = os.path.join(output_path, 'class_distribution.png')
    plt.savefig(dist_file)
    logger.info(f"Class distribution visualization saved to {dist_file}")
    plt.close()

def visualize_sample(df, output_path):
    """
    Visualize landmark data for a random sample.
    
    Args:
        df: DataFrame containing metadata
        output_path: Path to save visualization
    """
    try:
        plt.figure(figsize=(15, 5))
        
        # Get a random sample
        sample = df.sample(1).iloc[0]
        logger.info(f"\nVisualizing sample from class: {sample['class']}")
        
        # Load landmark data
        landmarks = np.load(sample['landmark_path'])
        
        # Check if we have multiple frames
        n_frames = min(3, landmarks.shape[0]) if landmarks.ndim > 1 else 1
        
        for i in range(n_frames):
            plt.subplot(1, n_frames, i+1)
            
            # Get landmarks for this frame
            frame_data = landmarks[i] if landmarks.ndim > 1 else landmarks
            
            # Try to interpret the data format
            if frame_data.size > 6:  # If we have enough points to represent landmarks
                # Create a blank image
                img = np.ones((400, 400, 3), dtype=np.uint8) * 255
                
                # Reshape if needed and extract x,y coordinates
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
                plt.title(f"Frame {i}")
            else:
                # Just plot the raw values
                plt.plot(frame_data)
                plt.title(f"Raw data for frame {i}")
            
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        visual_file = os.path.join(output_path, 'sample_visualization.png')
        plt.savefig(visual_file)
        logger.info(f"Sample visualization saved to {visual_file}")
        plt.close()
            
    except Exception as e:
        logger.error(f"Could not visualize data: {e}")
        logger.error("You may need to adjust the visualization code based on your data format.")

def main(args):
    """Main function to orchestrate the preprocessing workflow."""
    # Create paths
    landmarks_path = os.path.join(args.output, 'landmarks')
    
    # Create directories
    create_directories(args.output, landmarks_path)
    
    # List classes
    classes = list_classes(args.input)
    
    # Process landmark data
    data = process_landmarks(args.input, landmarks_path, classes)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save metadata
    metadata_path = os.path.join(args.output, 'metadata.csv')
    df.to_csv(metadata_path, index=False)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Analyze dataset
    analyze_dataset(df)
    
    # Generate visualizations
    visualize_class_distribution(df, args.output)
    
    # Visualize sample if requested
    if args.visualize:
        visualize_sample(df, args.output)
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)