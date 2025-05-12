#!/usr/bin/env python3
"""
Evaluation script for BISINDO Sign Language Recognition model.

This script evaluates a trained model on BISINDO sign language data, generating
detailed metrics, visualizations, and analyses of model performance.

Usage:
    python evaluate.py --data /path/to/processed/data --model /path/to/model/directory
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('evaluate')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate BISINDO Sign Language Recognition model')
    parser.add_argument('--data', type=str, default='../data/processed',
                        help='Path to the processed data directory')
    parser.add_argument('--model', type=str, default='../models',
                        help='Path to the directory containing the trained model')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing (0.0-1.0)')
    parser.add_argument('--output', type=str, default='../evaluation',
                        help='Path to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate and save visualizations')
    return parser.parse_args()

def load_data(dataframe, max_frames=100, num_landmarks=33, num_dimensions=3):
    """
    Load and preprocess landmark data from files referenced in the dataframe.
    
    Args:
        dataframe: Pandas DataFrame containing metadata about the samples
        max_frames: Maximum number of frames to keep (for padding/truncating)
        num_landmarks: Number of landmarks in each frame
        num_dimensions: Number of dimensions for each landmark (usually 3 for x,y,z)
        
    Returns:
        X: Array of landmark features
        y: Array of class labels
        sample_ids: List of sample identifiers
    """
    X = []
    y = []
    sample_ids = []
    
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Loading evaluation data"):
        try:
            # Load landmarks data
            landmarks = np.load(row['landmark_path'])
            
            # Handle different possible data shapes
            if landmarks.ndim == 1:
                # If data is flat, attempt to reshape
                landmarks = landmarks.reshape(1, -1)
            
            # Ensure landmarks have consistent dimensions
            if landmarks.shape[1] != num_landmarks * num_dimensions:
                # If dimensions don't match, reshape or skip
                if landmarks.shape[1] % num_dimensions == 0:
                    new_num_landmarks = landmarks.shape[1] // num_dimensions
                    landmarks = landmarks.reshape(landmarks.shape[0], new_num_landmarks, num_dimensions)
                else:
                    logger.warning(f"Skipping sample with incompatible shape: {landmarks.shape}")
                    continue
            else:
                # Reshape to (frames, landmarks, dimensions)
                landmarks = landmarks.reshape(landmarks.shape[0], num_landmarks, num_dimensions)
            
            # Normalize coordinates to [0, 1] range
            for i in range(num_dimensions):
                min_val = np.min(landmarks[:, :, i])
                max_val = np.max(landmarks[:, :, i])
                if max_val > min_val:
                    landmarks[:, :, i] = (landmarks[:, :, i] - min_val) / (max_val - min_val)
            
            # Handle sequence length (padding or truncating)
            if landmarks.shape[0] > max_frames:
                # Truncate if too long
                landmarks = landmarks[:max_frames]
            elif landmarks.shape[0] < max_frames:
                # Pad with zeros if too short
                padding = np.zeros((max_frames - landmarks.shape[0], landmarks.shape[1], landmarks.shape[2]))
                landmarks = np.vstack([landmarks, padding])
            
            # Reshape to the format expected by the model
            landmarks = landmarks.reshape(max_frames, -1)  # Flatten landmarks and dimensions
            
            X.append(landmarks)
            y.append(row['class_encoded'])
            sample_ids.append(row['sample_id'])
            
        except Exception as e:
            logger.error(f"Error processing {row['landmark_path']}: {e}")
    
    return np.array(X), np.array(y), sample_ids

def plot_confusion_matrix(cm, classes, output_path, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Function to plot and save the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Args:
        cm: Confusion matrix
        classes: List of class names
        output_path: Path to save the visualization
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Color map
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        normalized_text = "Normalized "
    else:
        fmt = 'd'
        normalized_text = ""
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(f'{normalized_text}{title}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    filename = f"{'normalized_' if normalize else ''}confusion_matrix.png"
    file_path = os.path.join(output_path, filename)
    plt.savefig(file_path)
    logger.info(f"Confusion matrix saved to {file_path}")
    plt.close()

def plot_class_metrics(class_metrics, output_path):
    """
    Plot and save class performance metrics.
    
    Args:
        class_metrics: DataFrame containing per-class metrics
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(14, 7))
    class_metrics.sort_values('F1 Score').plot(x='Class', y=['Precision', 'Recall', 'F1 Score'], 
                                              kind='bar', figsize=(14, 7))
    plt.title('Per-Class Performance Metrics')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the figure
    file_path = os.path.join(output_path, 'class_performance.png')
    plt.savefig(file_path)
    logger.info(f"Class performance metrics visualization saved to {file_path}")
    plt.close()

def plot_confidence_distribution(y_pred_prob, y_pred, y_test, output_path):
    """
    Plot and save confidence distribution for correct and incorrect predictions.
    
    Args:
        y_pred_prob: Predicted probabilities
        y_pred: Predicted class indices
        y_test: True class indices
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    confidence_correct = [max(prob) for prob, pred, true in zip(y_pred_prob, y_pred, y_test) if pred == true]
    confidence_incorrect = [max(prob) for prob, pred, true in zip(y_pred_prob, y_pred, y_test) if pred != true]

    plt.hist(confidence_correct, alpha=0.7, label='Correct predictions', bins=20, color='green')
    plt.hist(confidence_incorrect, alpha=0.7, label='Incorrect predictions', bins=20, color='red')
    plt.title('Model Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    file_path = os.path.join(output_path, 'confidence_distribution.png')
    plt.savefig(file_path)
    logger.info(f"Confidence distribution visualization saved to {file_path}")
    plt.close()

def plot_class_distribution(y_test, class_names, output_path):
    """
    Plot and save class distribution in the test dataset.
    
    Args:
        y_test: Test class indices
        class_names: List of class names
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    pd.Series(y_test).map(lambda x: class_names[x]).value_counts().sort_index().plot(kind='bar')
    plt.title('Class Distribution in Test Dataset')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    file_path = os.path.join(output_path, 'test_class_distribution.png')
    plt.savefig(file_path)
    logger.info(f"Test class distribution visualization saved to {file_path}")
    plt.close()

def analyze_misclassifications(y_test, y_pred, y_pred_prob, class_names, sample_ids, output_path):
    """
    Analyze and save information about misclassified samples.
    
    Args:
        y_test: True class indices
        y_pred: Predicted class indices
        y_pred_prob: Predicted probabilities
        class_names: List of class names
        sample_ids: List of sample IDs
        output_path: Path to save results
    """
    misclassified_indices = np.where(y_pred != y_test)[0]
    error_rate = len(misclassified_indices) / len(y_test) * 100
    
    logger.info(f"Total misclassified samples: {len(misclassified_indices)} out of {len(y_test)} ({error_rate:.2f}%)")
    
    if len(misclassified_indices) > 0:
        misclassified_data = []
        for idx in misclassified_indices:
            true_class = class_names[y_test[idx]]
            pred_class = class_names[y_pred[idx]]
            confidence = y_pred_prob[idx][y_pred[idx]]
            sample_id = sample_ids[idx]
            
            misclassified_data.append({
                'Sample ID': sample_id,
                'True Class': true_class,
                'Predicted Class': pred_class,
                'Confidence': confidence
            })
        
        misclassified_df = pd.DataFrame(misclassified_data)
        
        # Save misclassified samples data
        misclassified_file = os.path.join(output_path, 'misclassified_samples.csv')
        misclassified_df.to_csv(misclassified_file, index=False)
        logger.info(f"Misclassified samples data saved to {misclassified_file}")
        
        # Show common confusions
        confusion_pairs = misclassified_df.groupby(['True Class', 'Predicted Class']).size().reset_index(name='Count')
        confusion_pairs = confusion_pairs.sort_values('Count', ascending=False)
        
        logger.info("\nTop confused class pairs:")
        logger.info(confusion_pairs.head(10))
        
        # High confidence mistakes analysis
        high_conf_mistakes = misclassified_df[misclassified_df['Confidence'] > 0.8].sort_values('Confidence', ascending=False)
        
        if len(high_conf_mistakes) > 0:
            logger.info("\nHigh confidence mistakes:")
            logger.info(high_conf_mistakes.head(10))
            
            # Save high confidence mistakes
            high_conf_file = os.path.join(output_path, 'high_confidence_mistakes.csv')
            high_conf_mistakes.to_csv(high_conf_file, index=False)
            logger.info(f"High confidence mistakes saved to {high_conf_file}")
        else:
            logger.info("\nNo high confidence mistakes found.")
    
    return error_rate

def main(args):
    """Main function to orchestrate the evaluation workflow."""
    # Create output directory for evaluation results
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Evaluation results will be saved to {args.output}")
    
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
    
    # Load metadata
    try:
        metadata_path = os.path.join(args.data, 'metadata.csv')
        df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with {len(df)} samples")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        logger.error(f"Make sure metadata file exists at {metadata_path}")
        sys.exit(1)
    
    # Initialize and fit label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    # Encode class labels
    df['class_encoded'] = label_encoder.transform(df['class'])
    
    # Set landmark parameters
    MAX_FRAMES = 100
    NUM_LANDMARKS = 33
    NUM_DIMENSIONS = 3
    
    # Prepare test data
    test_df = df.sample(frac=args.test_size, random_state=42)
    logger.info(f"Using {len(test_df)} samples for evaluation (test_size={args.test_size})")
    
    # Load and preprocess test data
    X_test, y_test, sample_ids = load_data(test_df, max_frames=MAX_FRAMES, 
                                           num_landmarks=NUM_LANDMARKS, 
                                           num_dimensions=NUM_DIMENSIONS)
    
    # Evaluate model
    logger.info("\n--- Overall Model Evaluation ---")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate and save visualizations if requested
    if args.visualize:
        plot_confusion_matrix(cm, class_names, args.output, normalize=False, title='Confusion Matrix')
        plot_confusion_matrix(cm, class_names, args.output, normalize=True, title='Confusion Matrix')
    
    # Generate classification report
    logger.info("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info(report)
    
    # Save classification report
    report_file = os.path.join(args.output, 'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_file}")
    
    # Calculate per-class metrics
    logger.info("\n--- Per-Class Performance Metrics ---")
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    
    # Organize metrics into a DataFrame
    class_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Support': support
    })
    logger.info(class_metrics.sort_values('F1 Score'))
    
    # Save metrics
    metrics_file = os.path.join(args.output, 'class_metrics.csv')
    class_metrics.to_csv(metrics_file, index=False)
    logger.info(f"Class metrics saved to {metrics_file}")
    
    # Plot class metrics if requested
    if args.visualize:
        plot_class_metrics(class_metrics, args.output)
    
    # Analyze misclassifications
    error_rate = analyze_misclassifications(y_test, y_pred, y_pred_prob, class_names, sample_ids, args.output)
    
    # Confidence-correctness correlation
    all_confidences = [max(prob) for prob in y_pred_prob]
    correctness = [1 if pred == true else 0 for pred, true in zip(y_pred, y_test)]
    correlation = np.corrcoef(all_confidences, correctness)[0, 1]
    logger.info(f"\nCorrelation between model confidence and correctness: {correlation:.4f}")
    
    # Additional visualizations if requested
    if args.visualize:
        plot_confidence_distribution(y_pred_prob, y_pred, y_test, args.output)
        plot_class_distribution(y_test, class_names, args.output)
    
    # Save summary results
    summary = {
        'Test Accuracy': test_accuracy,
        'Test Loss': test_loss,
        'Error Rate': error_rate,
        'Confidence-Correctness Correlation': correlation,
        'Number of Test Samples': len(y_test),
        'Number of Classes': len(class_names)
    }
    
    summary_file = os.path.join(args.output, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Evaluation summary saved to {summary_file}")
    
    logger.info("\nEvaluation completed successfully!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)