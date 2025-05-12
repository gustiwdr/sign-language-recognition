import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

class ModelEvaluator:
    """
    Utility class for evaluating sign language recognition models.
    
    This class provides methods for calculating various evaluation metrics
    and generating visualizations to analyze model performance.
    """
    
    def __init__(self, class_names=None):
        """
        Initialize the ModelEvaluator with class names.
        
        Args:
            class_names: Array or list of class names
        """
        self.class_names = class_names
        
    def set_class_names(self, class_names):
        """
        Set class names after initialization.
        
        Args:
            class_names: Array or list of class names
        """
        self.class_names = class_names
        
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, title='Confusion Matrix', 
                              cmap=plt.cm.Blues, figsize=(12, 10), save_path=None):
        """
        Plot and optionally save a confusion matrix.
        
        Args:
            y_true: True label indices
            y_pred: Predicted label indices
            normalize: Whether to normalize the confusion matrix
            title: Plot title
            cmap: Colormap for the plot
            figsize: Figure size
            save_path: Path to save the figure (if None, the figure is just displayed)
            
        Returns:
            The confusion matrix array
        """
        if self.class_names is None:
            raise ValueError("Class names not set. Please set class names before plotting.")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            norm_text = "Normalized "
        else:
            fmt = 'd'
            norm_text = ""
        
        # Create figure
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{norm_text}{title}', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        return cm
    
    def get_classification_report(self, y_true, y_pred, output_dict=False):
        """
        Generate a classification report.
        
        Args:
            y_true: True label indices
            y_pred: Predicted label indices
            output_dict: Whether to return the report as a dictionary
            
        Returns:
            Classification report as string or dictionary
        """
        if self.class_names is None:
            raise ValueError("Class names not set. Please set class names before generating report.")
            
        return classification_report(y_true, y_pred, target_names=self.class_names, output_dict=output_dict)
    
    def calculate_per_class_metrics(self, y_true, y_pred):
        """
        Calculate precision, recall, and F1 score for each class.
        
        Args:
            y_true: True label indices
            y_pred: Predicted label indices
            
        Returns:
            DataFrame with per-class metrics
        """
        if self.class_names is None:
            raise ValueError("Class names not set. Please set class names before calculating metrics.")
            
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
        
        class_metrics = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Support': support
        })
        
        return class_metrics
    
    def plot_class_metrics(self, class_metrics, figsize=(14, 7), save_path=None):
        """
        Plot class performance metrics.
        
        Args:
            class_metrics: DataFrame from calculate_per_class_metrics
            figsize: Figure size
            save_path: Path to save the figure (if None, the figure is just displayed)
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
            plt.close()
        else:
            plt.show()
    
    def analyze_misclassifications(self, y_true, y_pred, y_pred_prob=None, sample_ids=None):
        """
        Analyze misclassified samples.
        
        Args:
            y_true: True label indices
            y_pred: Predicted label indices
            y_pred_prob: Prediction probabilities (optional)
            sample_ids: List of sample identifiers (optional)
            
        Returns:
            DataFrame with misclassification analysis and error rate
        """
        if self.class_names is None:
            raise ValueError("Class names not set. Please set class names before analysis.")
            
        misclassified_indices = np.where(y_pred != y_true)[0]
        error_rate = len(misclassified_indices) / len(y_true) * 100
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found.")
            return None, error_rate
        
        misclassified_data = []
        for idx in misclassified_indices:
            data = {
                'True Class': self.class_names[y_true[idx]],
                'Predicted Class': self.class_names[y_pred[idx]]
            }
            
            if y_pred_prob is not None:
                data['Confidence'] = float(y_pred_prob[idx][y_pred[idx]])
                
            if sample_ids is not None:
                data['Sample ID'] = sample_ids[idx]
                
            misclassified_data.append(data)
        
        misclassified_df = pd.DataFrame(misclassified_data)
        return misclassified_df, error_rate
    
    def get_confusion_pairs(self, misclassified_df, top_n=10):
        """
        Get the most common confusion pairs.
        
        Args:
            misclassified_df: DataFrame from analyze_misclassifications
            top_n: Number of top pairs to return
            
        Returns:
            DataFrame with confusion pairs sorted by frequency
        """
        if 'True Class' not in misclassified_df.columns or 'Predicted Class' not in misclassified_df.columns:
            raise ValueError("Misclassified DataFrame must contain 'True Class' and 'Predicted Class' columns.")
            
        confusion_pairs = misclassified_df.groupby(['True Class', 'Predicted Class']).size().reset_index(name='Count')
        return confusion_pairs.sort_values('Count', ascending=False).head(top_n)
    
    def plot_confidence_distribution(self, y_pred_prob, y_pred, y_true, figsize=(10, 6), save_path=None):
        """
        Plot confidence distribution for correct and incorrect predictions.
        
        Args:
            y_pred_prob: Prediction probabilities
            y_pred: Predicted label indices
            y_true: True label indices
            figsize: Figure size
            save_path: Path to save the figure (if None, the figure is just displayed)
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
            plt.close()
        else:
            plt.show()
    
    def calculate_confidence_correlation(self, y_pred_prob, y_pred, y_true):
        """
        Calculate correlation between model confidence and prediction correctness.
        
        Args:
            y_pred_prob: Prediction probabilities
            y_pred: Predicted label indices
            y_true: True label indices
            
        Returns:
            Correlation coefficient
        """
        all_confidences = [max(prob) for prob in y_pred_prob]
        correctness = [1 if pred == true else 0 for pred, true in zip(y_pred, y_true)]
        correlation = np.corrcoef(all_confidences, correctness)[0, 1]
        return correlation
    
    def plot_class_distribution(self, y, figsize=(12, 6), save_path=None):
        """
        Plot distribution of classes in a dataset.
        
        Args:
            y: Class indices
            figsize: Figure size
            save_path: Path to save the figure (if None, the figure is just displayed)
        """
        if self.class_names is None:
            raise ValueError("Class names not set. Please set class names before plotting.")
            
        plt.figure(figsize=figsize)
        pd.Series(y).map(lambda x: self.class_names[x]).value_counts().sort_index().plot(kind='bar')
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def get_evaluator(class_names=None):
    """
    Convenience function to quickly get a model evaluator.
    
    Args:
        class_names: Array or list of class names
        
    Returns:
        Configured ModelEvaluator instance
    """
    return ModelEvaluator(class_names)