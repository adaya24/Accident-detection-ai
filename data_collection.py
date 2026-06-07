"""
Data Collection Module for Accident Detection System
Collects and prepares traffic video data for model training
"""

import os
import cv2
import json
from datetime import datetime
from pathlib import Path

class DataCollector:
    """Handles collection and organization of traffic video data"""
    
    def __init__(self, dataset_dir="datasets"):
        """Initialize data collector with dataset directory"""
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True)
        self.raw_dir = self.dataset_dir / "raw_videos"
        self.processed_dir = self.dataset_dir / "processed_frames"
        self.annotations_dir = self.dataset_dir / "annotations"
        
        # Create subdirectories
        for dir_path in [self.raw_dir, self.processed_dir, self.annotations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def collect_video_from_source(self, source=0, duration=30, output_name="traffic_sample"):
        """
        Collect video from webcam or video file
        
        Args:
            source: Video source (0 for webcam, or file path)
            duration: Recording duration in seconds
            output_name: Name of output video file
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = self.raw_dir / f"{output_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        target_frames = duration * fps
        
        print(f"Recording {duration}s video to {output_path}...")
        
        while frame_count < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % fps == 0:
                print(f"Recorded {frame_count // fps}/{duration} seconds")
        
        cap.release()
        out.release()
        print(f"Video saved: {output_path}")
        return output_path
    
    def extract_frames(self, video_path, output_subdir="frames", sample_rate=5):
        """
        Extract frames from video for annotation
        
        Args:
            video_path: Path to video file
            output_subdir: Subdirectory name for frames
            sample_rate: Extract every nth frame
        """
        video_name = Path(video_path).stem
        frame_dir = self.processed_dir / video_name / output_subdir
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        extracted_count = 0
        
        print(f"Extracting frames from {video_path}...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frame_path = frame_dir / f"frame_{extracted_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {extracted_count} frames to {frame_dir}")
        return frame_dir
    
    def create_annotation_template(self, video_name):
        """Create annotation template for manual labeling"""
        annotation_file = self.annotations_dir / f"{video_name}_annotations.json"
        
        template = {
            "video": video_name,
            "created_at": datetime.now().isoformat(),
            "frames": [
                {
                    "frame_id": 0,
                    "has_collision": False,
                    "vehicle_count": 0,
                    "vehicles": [
                        {
                            "id": 0,
                            "class": "car",
                            "bbox": [x1, y1, x2, y2],
                            "confidence": 0.95
                        }
                    ],
                    "notes": ""
                }
            ],
            "statistics": {
                "total_frames": 0,
                "collision_frames": 0,
                "annotator": ""
            }
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Annotation template created: {annotation_file}")
        return annotation_file
    
    def get_dataset_stats(self):
        """Get statistics about collected dataset"""
        stats = {
            "raw_videos": len(list(self.raw_dir.glob("*.mp4"))),
            "processed_frames": len(list(self.processed_dir.glob("**/*.jpg"))),
            "annotations": len(list(self.annotations_dir.glob("*.json"))),
            "raw_size_mb": sum(f.stat().st_size for f in self.raw_dir.glob("*.mp4")) / (1024**2),
        }
        return stats


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Print dataset statistics
    stats = collector.get_dataset_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
