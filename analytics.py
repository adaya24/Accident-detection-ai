"""
Analytics and Monitoring Module
Tracks system performance and accident detection statistics
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('accident_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    """Tracks and analyzes system performance metrics"""
    
    def __init__(self, analytics_file="analytics_data.json"):
        """Initialize analytics tracker"""
        self.analytics_file = Path(analytics_file)
        self.metrics = defaultdict(list)
        self.load_existing_data()
    
    def log_detection_event(self, event_type, details):
        """
        Log a detection event
        
        Args:
            event_type: Type of event ('collision_detected', 'sos_sent', etc.)
            details: Dictionary with event details
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'details': details
        }
        
        self.metrics[event_type].append(event)
        logger.info(f"Event logged: {event_type} - {details}")
        
        return event
    
    def log_performance_metric(self, metric_name, value):
        """
        Log performance metric
        
        Args:
            metric_name: Name of metric (e.g., 'detection_latency_ms')
            value: Metric value
        """
        metric = {
            'timestamp': datetime.now().isoformat(),
            'metric': metric_name,
            'value': value
        }
        
        self.metrics['performance'].append(metric)
        logger.debug(f"Metric: {metric_name} = {value}")
    
    def log_model_inference(self, frame_count, inference_time_ms, detections):
        """
        Log model inference details
        
        Args:
            frame_count: Frame number processed
            inference_time_ms: Inference time in milliseconds
            detections: Number of objects detected
        """
        inference = {
            'timestamp': datetime.now().isoformat(),
            'frame': frame_count,
            'inference_time_ms': inference_time_ms,
            'detections': detections,
            'fps': 1000 / inference_time_ms if inference_time_ms > 0 else 0
        }
        
        self.metrics['inference'].append(inference)
    
    def get_statistics(self):
        """Generate statistics from collected metrics"""
        stats = {
            'total_collisions_detected': len(self.metrics.get('collision_detected', [])),
            'total_sos_alerts': len(self.metrics.get('sos_sent', [])),
            'average_detection_latency': self._calculate_avg_metric('detection_latency_ms'),
            'system_uptime_hours': self._calculate_uptime(),
            'total_events': sum(len(v) for v in self.metrics.values()),
            'generated_at': datetime.now().isoformat()
        }
        return stats
    
    def get_hourly_report(self):
        """Generate hourly collision statistics"""
        hourly_collisions = defaultdict(int)
        
        for event in self.metrics.get('collision_detected', []):
            timestamp = datetime.fromisoformat(event['timestamp'])
            hour_key = timestamp.strftime("%Y-%m-%d %H:00")
            hourly_collisions[hour_key] += 1
        
        return dict(hourly_collisions)
    
    def _calculate_avg_metric(self, metric_name):
        """Calculate average value for a metric"""
        values = [m['value'] for m in self.metrics.get('performance', [])
                 if m['metric'] == metric_name]
        return sum(values) / len(values) if values else 0
    
    def _calculate_uptime(self):
        """Calculate system uptime"""
        if not self.metrics.get('inference'):
            return 0
        
        first_event = self.metrics['inference'][0]
        last_event = self.metrics['inference'][-1]
        
        start = datetime.fromisoformat(first_event['timestamp'])
        end = datetime.fromisoformat(last_event['timestamp'])
        
        return (end - start).total_seconds() / 3600
    
    def save_analytics(self):
        """Save analytics data to file"""
        data = {
            'statistics': self.get_statistics(),
            'hourly_report': self.get_hourly_report(),
            'metrics': {k: v for k, v in self.metrics.items()}
        }
        
        with open(self.analytics_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Analytics saved to {self.analytics_file}")
    
    def load_existing_data(self):
        """Load existing analytics data"""
        if self.analytics_file.exists():
            try:
                with open(self.analytics_file, 'r') as f:
                    data = json.load(f)
                    if 'metrics' in data:
                        self.metrics = defaultdict(list, data['metrics'])
                logger.info("Loaded existing analytics data")
            except Exception as e:
                logger.error(f"Error loading analytics: {e}")
    
    def generate_html_dashboard(self, output_file="analytics_dashboard.html"):
        """Generate HTML dashboard for analytics"""
        stats = self.get_statistics()
        hourly = self.get_hourly_report()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Accident Detection Analytics Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 900px; margin: auto; background: white; padding: 20px; border-radius: 8px; }}
                h1 {{ color: #dc3545; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f9f9f9; border-radius: 4px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #dc3545; }}
                .metric-label {{ color: #666; font-size: 12px; }}
                table {{ width: 100%; margin-top: 20px; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #dc3545; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚨 Accident Detection System Analytics</h1>
                
                <h2>Key Metrics</h2>
                <div class="metric">
                    <div class="metric-label">Total Collisions</div>
                    <div class="metric-value">{stats['total_collisions_detected']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">SOS Alerts Sent</div>
                    <div class="metric-value">{stats['total_sos_alerts']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Detection Latency</div>
                    <div class="metric-value">{stats['average_detection_latency']:.0f}ms</div>
                </div>
                <div class="metric">
                    <div class="metric-label">System Uptime</div>
                    <div class="metric-value">{stats['system_uptime_hours']:.1f}h</div>
                </div>
                
                <h2>Hourly Collision Summary</h2>
                <table>
                    <tr>
                        <th>Hour</th>
                        <th>Collisions Detected</th>
                    </tr>
        """
        
        for hour, count in sorted(hourly.items()):
            html_content += f"<tr><td>{hour}</td><td>{count}</td></tr>"
        
        html_content += """
                </table>
                
                <p style="color: #999; font-size: 12px; margin-top: 30px;">
                    Report generated at """ + stats['generated_at'] + """
                </p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard generated: {output_file}")


if __name__ == "__main__":
    # Example usage
    analytics = PerformanceAnalytics()
    print("Analytics module initialized")
    print("Use this to track system performance and accident statistics")
