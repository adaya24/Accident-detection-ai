"""
Fine-Tuning Module for Custom Accident Detection Model
Trains YOLOv8 on custom accident-specific dataset
"""

import json
from pathlib import Path
from ultralytics import YOLO
import yaml

class ModelFineTuner:
    """Fine-tunes YOLOv8 model on custom accident detection data"""
    
    def __init__(self, base_model="yolov8n.pt", project_dir="runs/detect"):
        """
        Initialize fine-tuner
        
        Args:
            base_model: Base YOLO model to fine-tune
            project_dir: Directory for training results
        """
        self.model = YOLO(base_model)
        self.project_dir = Path(project_dir)
        self.training_history = []
    
    def prepare_dataset_yaml(self, data_path, train_dir, val_dir, test_dir=None):
        """
        Prepare YAML configuration for YOLO training
        
        Args:
            data_path: Directory to save dataset.yaml
            train_dir: Path to training images
            val_dir: Path to validation images
            test_dir: Path to test images (optional)
        """
        dataset_config = {
            'path': str(Path(data_path).absolute()),
            'train': str(Path(train_dir).relative_to(data_path)),
            'val': str(Path(val_dir).relative_to(data_path)),
            'nc': 2,  # 2 classes: collision, no_collision
            'names': ['collision', 'normal']
        }
        
        if test_dir:
            dataset_config['test'] = str(Path(test_dir).relative_to(data_path))
        
        yaml_path = Path(data_path) / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f)
        
        print(f"Dataset YAML created: {yaml_path}")
        return yaml_path
    
    def train(self, data_yaml, epochs=50, imgsz=640, batch_size=16, patience=20):
        """
        Fine-tune model on custom dataset
        
        Args:
            data_yaml: Path to dataset.yaml
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Batch size for training
            patience: Early stopping patience
        
        Returns:
            Training results object
        """
        print(f"Starting fine-tuning with {epochs} epochs...")
        print(f"Dataset: {data_yaml}")
        
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            project=str(self.project_dir),
            device=0,  # GPU device, set to -1 for CPU
            verbose=True,
            save=True,
            cache=True
        )
        
        self.training_history.append({
            'epochs': epochs,
            'batch_size': batch_size,
            'imgsz': imgsz,
            'results': str(results.save_dir)
        })
        
        print(f"Training completed. Results saved to {results.save_dir}")
        return results
    
    def validate(self, data_yaml):
        """
        Validate fine-tuned model
        
        Args:
            data_yaml: Path to dataset.yaml
        
        Returns:
            Validation metrics
        """
        print("Running validation...")
        metrics = self.model.val(data=str(data_yaml))
        return metrics
    
    def save_model(self, output_path="accident_detector_finetuned.pt"):
        """
        Save fine-tuned model
        
        Args:
            output_path: Path to save model
        """
        self.model.save(output_path)
        print(f"Model saved: {output_path}")
    
    def get_training_summary(self):
        """Get summary of training history"""
        return {
            'total_training_runs': len(self.training_history),
            'history': self.training_history
        }


if __name__ == "__main__":
    # Example usage
    print("Fine-tuning module for custom accident detection model")
    print("Usage:")
    print("  1. Prepare your dataset in train/val/test directories")
    print("  2. Create dataset.yaml with class definitions")
    print("  3. Run fine-tuning with prepared dataset")
