import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import os

class LandmarkVisualizer:
    """
    Utility class for visualizing landmark data from the BISINDO sign language dataset.
    
    This class provides methods to visualize landmark data in various formats,
    including spatial visualization of landmarks and sequence visualizations.
    """
    
    def __init__(self, output_path=None):
        """
        Initialize the LandmarkVisualizer.
        
        Args:
            output_path: Directory to save visualizations (optional)
        """
        self.output_path = output_path
        
    def visualize_frame(self, frame_data, title="Landmark Frame", figsize=(6, 6), 
                         save_path=None, show=True):
        """
        Visualize a single frame of landmark data.
        
        Args:
            frame_data: Landmark data for a single frame
            title: Plot title
            figsize: Figure size
            save_path: Path to save the visualization
            show: Whether to display the plot
            
        Returns:
            Image array of the visualization
        """
        plt.figure(figsize=figsize)
        
        # Create a blank image
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Process landmark data
        if frame_data.ndim > 1 and frame_data.shape[1] >= 2:
            # If we have multi-dimensional data with at least x,y columns
            points = frame_data[:, :2]  # Take first two dimensions (x,y)
        elif frame_data.size > 6:
            # If flat array, try to reshape assuming groups of 3 (x,y,z)
            points = frame_data.reshape(-1, 3)[:, :2]
        else:
            # If we can't reshape appropriately, just plot the raw values
            plt.plot(frame_data)
            plt.title(title)
            
            if save_path:
                plt.savefig(save_path)
            
            if show:
                plt.show()
            else:
                plt.close()
                
            return img
        
        # Normalize points to fit in image
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        range_vals = max_vals - min_vals
        
        if np.any(range_vals):
            normalized_points = (points - min_vals) / range_vals
        else:
            normalized_points = points
        
        # Plot points
        for pt in normalized_points:
            x, y = int(pt[0] * 350 + 25), int(pt[1] * 350 + 25)
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return img
    
    def visualize_sample(self, landmarks, class_name=None, max_frames=3, figsize=(15, 5), 
                          save_path=None, show=True):
        """
        Visualize a sample with multiple frames.
        
        Args:
            landmarks: Landmark data array
            class_name: Class name for the sample
            max_frames: Maximum number of frames to display
            figsize: Figure size
            save_path: Path to save the visualization
            show: Whether to display the plot
            
        Returns:
            List of image arrays
        """
        plt.figure(figsize=figsize)
        
        # Check if we have multiple frames
        n_frames = min(max_frames, landmarks.shape[0]) if landmarks.ndim > 1 else 1
        
        images = []
        for i in range(n_frames):
            plt.subplot(1, n_frames, i+1)
            
            # Get landmarks for this frame
            frame_data = landmarks[i] if landmarks.ndim > 1 else landmarks
            
            # Create title
            frame_title = f"Frame {i}" if not class_name else f"{class_name} - Frame {i}"
            
            # Visualize frame (without showing)
            img = self.visualize_frame(frame_data, title=frame_title, show=False)
            images.append(img)
            
            plt.title(frame_title)
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return images
    
    def visualize_random_samples(self, dataframe, n_samples=5, max_frames=3, 
                                  figsize=(15, 5), save_dir=None):
        """
        Visualize random samples from the dataset.
        
        Args:
            dataframe: DataFrame containing metadata
            n_samples: Number of random samples to visualize
            max_frames: Maximum number of frames per sample
            figsize: Figure size
            save_dir: Directory to save visualizations
            
        Returns:
            None
        """
        for i in range(n_samples):
            # Get a random sample
            sample = dataframe.sample(1).iloc[0]
            
            # Load landmark data
            try:
                landmarks = np.load(sample['landmark_path'])
                
                print(f"\nVisualizing sample {i+1}/{n_samples} from class: {sample['class']}")
                
                # Create save path if needed
                save_path = None
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"sample_{sample['sample_id']}.png")
                
                # Visualize sample
                self.visualize_sample(
                    landmarks, 
                    class_name=sample['class'], 
                    max_frames=max_frames,
                    figsize=figsize,
                    save_path=save_path
                )
                
            except Exception as e:
                print(f"Error visualizing sample {sample['sample_id']}: {e}")


class ModelVisualizer:
    """
    Utility class for visualizing model training and evaluation results.
    
    This class provides methods to create various plots related to model performance,
    including training history, confusion matrices, and class performance metrics.
    """
    
    def __init__(self, output_path=None):
        """
        Initialize the ModelVisualizer.
        
        Args:
            output_path: Directory to save visualizations (optional)
        """
        self.output_path = output_path
        if output_path:
            os.makedirs(output_path, exist_ok=True)
    
    def plot_training_history(self, history, figsize=(12, 5), save_path=None):
        """
        Plot training history metrics.
        
        Args:
            history: Training history object from model.fit()
            figsize: Figure size
            save_path: Path to save the visualization
            
        Returns:
            None
        """
        plt.figure(figsize=figsize)
        
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
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names, normalize=False, 
                              title='Confusion Matrix', cmap=plt.cm.Blues,
                              figsize=(12, 10), save_path=None):
        """
        Plot a confusion matrix.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize the confusion matrix
            title: Plot title
            cmap: Colormap
            figsize: Figure size
            save_path: Path to save the visualization
            
        Returns:
            None
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            norm_text = "Normalized "
        else:
            fmt = 'd'
            norm_text = ""
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{norm_text}{title}', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_class_metrics(self, class_metrics, figsize=(14, 7), save_path=None):
        """
        Plot class performance metrics.
        
        Args:
            class_metrics: DataFrame with class metrics
            figsize: Figure size
            save_path: Path to save the visualization
            
        Returns:
            None
        """
        plt.figure(figsize=figsize)
        class_metrics.sort_values('F1 Score').plot(x='Class', y=['Precision', 'Recall', 'F1 Score'], 
                                                  kind='bar', figsize=figsize)
        plt.title('Per-Class Performance Metrics')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Class metrics plot saved to {save_path}")
        
        plt.show()
    
    def plot_confidence_distribution(self, y_pred_prob, y_pred, y_true, 
                                    figsize=(10, 6), save_path=None):
        """
        Plot confidence distribution for correct and incorrect predictions.
        
        Args:
            y_pred_prob: Predicted probabilities
            y_pred: Predicted classes
            y_true: True classes
            figsize: Figure size
            save_path: Path to save the visualization
            
        Returns:
            None
        """
        confidence_correct = [max(prob) for prob, pred, true in zip(y_pred_prob, y_pred, y_true) if pred == true]
        confidence_incorrect = [max(prob) for prob, pred, true in zip(y_pred_prob, y_pred, y_true) if pred != true]

        plt.figure(figsize=figsize)
        plt.hist(confidence_correct, alpha=0.7, label='Correct predictions', bins=20, color='green')
        plt.hist(confidence_incorrect, alpha=0.7, label='Incorrect predictions', bins=20, color='red')
        plt.title('Model Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confidence distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, y, class_names, figsize=(12, 6), save_path=None):
        """
        Plot distribution of classes in a dataset.
        
        Args:
            y: Class indices
            class_names: List of class names
            figsize: Figure size
            save_path: Path to save the visualization
            
        Returns:
            None
        """
        plt.figure(figsize=figsize)
        pd.Series(y).map(lambda x: class_names[x]).value_counts().sort_index().plot(kind='bar')
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()
    
    def visualize_prediction(self, landmark_data, prediction, class_names, true_class=None,
                              figsize=(15, 10), save_path=None):
        """
        Visualize a model prediction alongside landmark data.
        
        Args:
            landmark_data: Landmark data for the sample
            prediction: Prediction probabilities
            class_names: List of class names
            true_class: True class (optional)
            figsize: Figure size
            save_path: Path to save the visualization
            
        Returns:
            None
        """
        plt.figure(figsize=figsize)
        
        # Get top predictions
        top_indices = np.argsort(prediction)[-5:][::-1]
        top_classes = [class_names[i] for i in top_indices]
        top_probs = [prediction[i] for i in top_indices]
        
        # Plot landmark visualization
        plt.subplot(2, 1, 1)
        
        # Select a representative frame from landmarks
        if landmark_data.ndim > 2:
            # If we have multiple frames, pick one in the middle
            frame_idx = min(10, landmark_data.shape[0] - 1)
            frame_data = landmark_data[frame_idx]
        else:
            frame_data = landmark_data
        
        # Create a blank image
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Extract and normalize points for visualization
        if frame_data.ndim > 1:
            points = frame_data[:, :2]
        else:
            points = frame_data.reshape(-1, 3)[:, :2]
        
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        range_vals = max_vals - min_vals
        
        if np.any(range_vals):
            normalized_points = (points - min_vals) / range_vals
        else:
            normalized_points = points
        
        # Plot points
        for pt in normalized_points:
            x, y = int(pt[0] * 350 + 25), int(pt[1] * 350 + 25)
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        
        plt.imshow(img)
        
        # Add title with true class if provided
        if true_class is not None:
            plt.title(f"True Class: {true_class}")
        else:
            plt.title("Input Landmarks")
        
        plt.axis('off')
        
        # Plot prediction probabilities
        plt.subplot(2, 1, 2)
        y_pos = np.arange(len(top_classes))
        plt.barh(y_pos, top_probs, align='center')
        plt.yticks(y_pos, top_classes)
        plt.xlabel('Probability')
        plt.title('Model Predictions')
        
        # Add a banner showing top prediction
        pred_class = class_names[np.argmax(prediction)]
        pred_prob = np.max(prediction)
        plt.figtext(0.5, 0.01, f"Predicted: {pred_class} (Confidence: {pred_prob:.2f})",
                   ha='center', fontsize=12, bbox={'facecolor': 'yellow', 'alpha': 0.5})
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Prediction visualization saved to {save_path}")
        
        plt.show()


def get_landmark_visualizer(output_path=None):
    """
    Convenience function to get a landmark visualizer.
    
    Args:
        output_path: Directory to save visualizations (optional)
        
    Returns:
        LandmarkVisualizer instance
    """
    return LandmarkVisualizer(output_path)

def get_model_visualizer(output_path=None):
    """
    Convenience function to get a model visualizer.
    
    Args:
        output_path: Directory to save visualizations (optional)
        
    Returns:
        ModelVisualizer instance
    """
    return ModelVisualizer(output_path)