import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DatasetLoader:
    """
    Utility class for loading and preprocessing BISINDO sign language dataset.
    
    This class handles loading the metadata and landmark data, preprocessing landmarks,
    encoding classes, and splitting the dataset for training, validation, and testing.
    """
    
    def __init__(self, processed_path='../data/processed', max_frames=100, 
                  num_landmarks=33, num_dimensions=3):
        """
        Initialize the DatasetLoader with dataset parameters.
        
        Args:
            processed_path: Path to the processed data directory
            max_frames: Maximum number of frames to keep per sample
            num_landmarks: Number of landmarks per frame
            num_dimensions: Number of dimensions per landmark (usually 3 for x,y,z)
        """
        self.processed_path = processed_path
        self.landmarks_path = os.path.join(processed_path, 'landmarks')
        self.max_frames = max_frames
        self.num_landmarks = num_landmarks
        self.num_dimensions = num_dimensions
        self.label_encoder = LabelEncoder()
        self.metadata = None
        self.classes = None
    
    def load_metadata(self):
        """
        Load metadata CSV file into a pandas DataFrame.
        
        Returns:
            DataFrame containing metadata
        """
        metadata_path = os.path.join(self.processed_path, 'metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Run preprocessing first.")
            
        self.metadata = pd.read_csv(metadata_path)
        return self.metadata
    
    def encode_classes(self):
        """
        Encode class labels using LabelEncoder.
        
        Returns:
            Updated DataFrame with encoded classes
        """
        if self.metadata is None:
            self.load_metadata()
            
        self.classes = self.metadata['class'].unique()
        self.label_encoder.fit(self.classes)
        self.metadata['class_encoded'] = self.label_encoder.transform(self.metadata['class'])
        return self.metadata
    
    def get_class_names(self):
        """
        Get class names from the label encoder.
        
        Returns:
            Array of class names
        """
        if self.classes is None:
            self.encode_classes()
        return self.label_encoder.classes_
    
    def preprocess_landmarks(self, landmarks):
        """
        Preprocess landmark data for model input.
        
        Args:
            landmarks: Raw landmark data array
            
        Returns:
            Preprocessed landmark data
        """
        # Handle different possible data shapes
        if landmarks.ndim == 1:
            # If data is flat, attempt to reshape
            landmarks = landmarks.reshape(1, -1)
        
        # Ensure landmarks have consistent dimensions
        if landmarks.shape[1] != self.num_landmarks * self.num_dimensions:
            # If dimensions don't match, reshape
            if landmarks.shape[1] % self.num_dimensions == 0:
                new_num_landmarks = landmarks.shape[1] // self.num_dimensions
                landmarks = landmarks.reshape(landmarks.shape[0], new_num_landmarks, self.num_dimensions)
            else:
                raise ValueError(f"Incompatible landmark shape: {landmarks.shape}")
        else:
            # Reshape to (frames, landmarks, dimensions)
            landmarks = landmarks.reshape(landmarks.shape[0], self.num_landmarks, self.num_dimensions)
        
        # Normalize coordinates to [0, 1] range
        for i in range(self.num_dimensions):
            min_val = np.min(landmarks[:, :, i])
            max_val = np.max(landmarks[:, :, i])
            if max_val > min_val:
                landmarks[:, :, i] = (landmarks[:, :, i] - min_val) / (max_val - min_val)
        
        # Handle sequence length (padding or truncating)
        if landmarks.shape[0] > self.max_frames:
            # Truncate if too long
            landmarks = landmarks[:self.max_frames]
        elif landmarks.shape[0] < self.max_frames:
            # Pad with zeros if too short
            padding = np.zeros((self.max_frames - landmarks.shape[0], landmarks.shape[1], landmarks.shape[2]))
            landmarks = np.vstack([landmarks, padding])
        
        # Reshape to the format expected by the model
        landmarks = landmarks.reshape(self.max_frames, -1)  # Flatten landmarks and dimensions
        
        return landmarks
    
    def load_data(self, include_sample_ids=False):
        """
        Load and preprocess landmark data from files referenced in the metadata.
        
        Args:
            include_sample_ids: Whether to include sample IDs in the output
            
        Returns:
            X: Array of landmark features
            y: Array of class labels
            sample_ids (optional): List of sample IDs
        """
        if self.metadata is None or 'class_encoded' not in self.metadata.columns:
            self.encode_classes()
            
        X = []
        y = []
        sample_ids = [] if include_sample_ids else None
        
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Loading data"):
            try:
                # Load landmarks data
                landmarks = np.load(row['landmark_path'])
                
                # Preprocess landmarks
                processed_landmarks = self.preprocess_landmarks(landmarks)
                
                # Add to dataset
                X.append(processed_landmarks)
                y.append(row['class_encoded'])
                
                if include_sample_ids:
                    sample_ids.append(row['sample_id'])
                    
            except Exception as e:
                print(f"Error processing {row['landmark_path']}: {e}")
        
        if include_sample_ids:
            return np.array(X), np.array(y), sample_ids
        else:
            return np.array(X), np.array(y)
    
    def split_dataset(self, test_size=0.2, val_size=0.15, random_state=42):
        """
        Load data and split into training, validation, and test sets.
        
        Args:
            test_size: Fraction of data to use for testing
            val_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, val, and test splits with their features and labels
        """
        # Load data
        X, y = self.load_data()
        
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation sets
        # Adjust val_size to be relative to the train_val set
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=relative_val_size, 
            random_state=random_state, stratify=y_train_val
        )
        
        return {
            'train': {'X': X_train, 'y': y_train},
            'validation': {'X': X_val, 'y': y_val},
            'test': {'X': X_test, 'y': y_test}
        }
    
    def load_test_data(self, test_size=0.2, random_state=42):
        """
        Load a subset of data for testing/evaluation with sample IDs.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_test: Test features
            y_test: Test labels
            sample_ids: Sample IDs for test data
        """
        if self.metadata is None or 'class_encoded' not in self.metadata.columns:
            self.encode_classes()
            
        # Sample a portion of the metadata for testing
        test_df = self.metadata.sample(frac=test_size, random_state=random_state)
        
        # Load data with sample IDs
        X_test, y_test, sample_ids = self.load_data_from_dataframe(test_df, include_sample_ids=True)
        
        return X_test, y_test, sample_ids
    
    def load_data_from_dataframe(self, dataframe, include_sample_ids=False):
        """
        Load and preprocess landmark data from files referenced in the given dataframe.
        
        Args:
            dataframe: DataFrame containing metadata about the samples
            include_sample_ids: Whether to include sample IDs in the output
            
        Returns:
            X: Array of landmark features
            y: Array of class labels
            sample_ids (optional): List of sample IDs
        """
        X = []
        y = []
        sample_ids = [] if include_sample_ids else None
        
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Loading data"):
            try:
                # Load landmarks data
                landmarks = np.load(row['landmark_path'])
                
                # Preprocess landmarks
                processed_landmarks = self.preprocess_landmarks(landmarks)
                
                # Add to dataset
                X.append(processed_landmarks)
                y.append(row['class_encoded'])
                
                if include_sample_ids:
                    sample_ids.append(row['sample_id'])
                    
            except Exception as e:
                print(f"Error processing {row['landmark_path']}: {e}")
        
        if include_sample_ids:
            return np.array(X), np.array(y), sample_ids
        else:
            return np.array(X), np.array(y)

def get_dataset(processed_path='../data/processed', max_frames=100, 
                num_landmarks=33, num_dimensions=3):
    """
    Convenience function to quickly get a dataset loader.
    
    Args:
        processed_path: Path to the processed data directory
        max_frames: Maximum number of frames to keep per sample
        num_landmarks: Number of landmarks per frame
        num_dimensions: Number of dimensions per landmark
        
    Returns:
        Configured DatasetLoader instance
    """
    loader = DatasetLoader(
        processed_path=processed_path,
        max_frames=max_frames,
        num_landmarks=num_landmarks,
        num_dimensions=num_dimensions
    )
    loader.load_metadata()
    loader.encode_classes()
    return loader