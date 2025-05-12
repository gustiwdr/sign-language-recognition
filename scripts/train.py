#!/usr/bin/env python3
"""
Training script for BISINDO Sign Language Recognition model.

This script trains a deep learning model on preprocessed BISINDO sign language landmark data.
It loads the data, builds a Conv1D-LSTM model, trains it, and saves the trained model
along with evaluation metrics.

Usage:
    python train.py --data /path/to/processed/data --output /path/to/models
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
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
logger = logging.getLogger('train')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BISINDO Sign Language Recognition model')
    parser.add_argument('--data', type=str, default='../data/processed',
                        help='Path to the processed data directory')
    parser.add_argument('--output', type=str, default='../models',
                        help='Path to save trained models')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
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
    """
    X = []
    y = []
    
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Loading data"):
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
            
        except Exception as e:
            logger.error(f"Error processing {row['landmark_path']}: {e}")
    
    return np.array(X), np.array(y)

def create_model(input_shape, num_classes):
    """
    Create a sequential model for sign language recognition.
    
    Args:
        input_shape: Shape of input data (frames, features)
        num_classes: Number of output classes
        
    Returns:
        Compiled model
    """
    model = Sequential([
        # First Conv1D layer
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Second Conv1D layer
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM layers for sequence modeling
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        
        # Output layer
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history, output_path):
    """
    Plot and save training history visualizations.
    
    Args:
        history: Training history object from model.fit()
        output_path: Directory to save visualizations
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    # Save visualization
    history_file = os.path.join(output_path, 'training_history.png')
    plt.savefig(history_file)
    logger.info(f"Training history plots saved to {history_file}")
    plt.close()

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot and save confusion matrix visualization.
    
    Args:
        cm: Confusion matrix array
        class_names: Array of class names
        output_path: Directory to save visualization
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save visualization
    cm_file = os.path.join(output_path, 'confusion_matrix.png')
    plt.savefig(cm_file)
    logger.info(f"Confusion matrix saved to {cm_file}")
    plt.close()

def main(args):
    """Main function to orchestrate the training workflow."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Model output directory: {args.output}")
    
    # Define paths
    processed_path = args.data
    landmarks_path = os.path.join(processed_path, 'landmarks')
    
    # Load metadata
    metadata_path = os.path.join(processed_path, 'metadata.csv')
    
    try:
        df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with {len(df)} samples")
        logger.info(f"Number of classes: {df['class'].nunique()}")
        logger.info(f"Classes: {df['class'].unique()}")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        logger.error(f"Make sure the metadata file exists at {metadata_path}")
        sys.exit(1)
    
    # Encode class labels
    label_encoder = LabelEncoder()
    df['class_encoded'] = label_encoder.fit_transform(df['class'])
    
    # Save class mappings
    np.save(os.path.join(args.output, 'classes.npy'), label_encoder.classes_)
    logger.info(f"Class mapping saved to {os.path.join(args.output, 'classes.npy')}")
    
    # Set landmark parameters
    MAX_FRAMES = 100
    NUM_LANDMARKS = 33
    NUM_DIMENSIONS = 3
    
    # Load and preprocess data
    X, y = load_data(df, max_frames=MAX_FRAMES, num_landmarks=NUM_LANDMARKS, num_dimensions=NUM_DIMENSIONS)
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    logger.info(f"Training set: {X_train.shape}, {y_train.shape}")
    logger.info(f"Validation set: {X_val.shape}, {y_val.shape}")
    logger.info(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Setup model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)
    
    model = create_model(input_shape, num_classes)
    model.summary(print_fn=logger.info)
    
    # Set up callbacks
    callbacks = [
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            filepath=os.path.join(args.output, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Stop training when validation loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(args.output, 'final_model.h5'))
    logger.info(f"Final model saved to {os.path.join(args.output, 'final_model.h5')}")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Save classification report
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred_classes, target_names=class_names)
    logger.info("\nClassification Report:")
    logger.info(report)
    
    with open(os.path.join(args.output, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Generate visualizations if requested
    if args.visualize:
        plot_training_history(history, args.output)
        plot_confusion_matrix(cm, class_names, args.output)
    
    logger.info("Model training completed successfully!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)