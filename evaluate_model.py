"""
Evaluation Pipeline for Accident Detection Model
Comprehensive testing and performance metrics calculation
"""

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

class ModelEvaluator:
    """Evaluates accident detection model performance"""
    
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize evaluator with model
        
        Args:
            model_path: Path to YOLO model
        """
        self.model = YOLO(model_path)
        self.metrics = defaultdict(list)
        self.results_log = []
    
    def evaluate_on_image(self, image_path, conf_threshold=0.5):
        """
        Evaluate model on single image
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
        
        Returns:
            Detection results
        """
        results = self.model(image_path, conf=conf_threshold, verbose=False)[0]
        return results
    
    def evaluate_on_video(self, video_path, output_path=None, conf_threshold=0.5):
        """
        Evaluate model on video frames
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save annotated video
            conf_threshold: Confidence threshold
        
        Returns:
            Frame-by-frame results and metrics
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_results = []
        collision_frames = 0
        total_frames = 0
        detection_times = []
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Evaluating video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames += 1
            
            # Run detection
            results = self.model(frame, conf=conf_threshold, verbose=False)[0]
            detection_times.append(results.speed['inference'])
            
            # Check for collision
            vehicle_boxes = []
            for r in results.boxes:
                cls = int(r.cls[0])
                if cls in [2, 3, 5, 7]:  # vehicle classes
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    vehicle_boxes.append((x1, y1, x2, y2))
            
            # Calculate IoU-based collision
            has_collision = self._check_collision(vehicle_boxes)
            if has_collision:
                collision_frames += 1
            
            frame_results.append({
                'frame': total_frames,
                'vehicles_detected': len(vehicle_boxes),
                'collision_detected': has_collision,
                'confidence': float(np.mean([b.conf[0] for b in results.boxes]) if results.boxes else 0)
            })
            
            # Annotate frame if output requested
            if output_path:
                annotated_frame = results.plot()
                if has_collision:
                    cv2.putText(annotated_frame, "COLLISION DETECTED", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(annotated_frame)
        
        cap.release()
        if output_path:
            out.release()
            print(f"Annotated video saved: {output_path}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            total_frames, collision_frames, detection_times
        )
        
        return {
            'frame_results': frame_results,
            'metrics': metrics
        }
    
    def _check_collision(self, vehicle_boxes, iou_threshold=0.3):
        """Check for collision using IoU"""
        if len(vehicle_boxes) < 2:
            return False
        
        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):
                iou = self._calculate_iou(vehicle_boxes[i], vehicle_boxes[j])
                if iou > iou_threshold:
                    return True
        return False
    
    def _calculate_iou(self, boxA, boxB):
        """Calculate Intersection over Union"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    
    def _calculate_metrics(self, total_frames, collision_frames, detection_times):
        """Calculate performance metrics"""
        metrics = {
            'total_frames': total_frames,
            'collision_frames': collision_frames,
            'collision_rate': collision_frames / total_frames if total_frames > 0 else 0,
            'avg_detection_time_ms': np.mean(detection_times),
            'max_detection_time_ms': np.max(detection_times),
            'min_detection_time_ms': np.min(detection_times),
            'fps': 1000 / np.mean(detection_times) if detection_times else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results_log.append(metrics)
        return metrics
    
    def get_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.results_log:
            return "No evaluation results available"
        
        latest = self.results_log[-1]
        report = f"""
        ===== ACCIDENT DETECTION MODEL EVALUATION REPORT =====
        
        Frames Analyzed: {latest['total_frames']}
        Collisions Detected: {latest['collision_frames']}
        Collision Detection Rate: {latest['collision_rate']:.2%}
        
        Performance Metrics:
        - Average Detection Time: {latest['avg_detection_time_ms']:.2f} ms
        - FPS Throughput: {latest['fps']:.1f}
        - Min Detection Time: {latest['min_detection_time_ms']:.2f} ms
        - Max Detection Time: {latest['max_detection_time_ms']:.2f} ms
        
        Evaluation Time: {latest['timestamp']}
        """
        return report
    
    def save_evaluation_results(self, output_file="evaluation_results.json"):
        """Save evaluation results to file"""
        with open(output_file, 'w') as f:
            json.dump(self.results_log, f, indent=2)
        print(f"Results saved: {output_file}")


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Pipeline")
    print("Evaluate pre-trained or fine-tuned models on test data")
