from abc import ABC, abstractmethod
import torch

class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self):
        """Load pre-trained model"""
        pass
    
    @abstractmethod
    def predict(self, text):
        """Make prediction on single text"""
        pass
    
    @abstractmethod
    def predict_batch(self, texts):
        """Make predictions on batch of texts"""
        pass
