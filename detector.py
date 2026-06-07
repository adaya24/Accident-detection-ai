from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def detect_collision(frame):
    results = model(frame, verbose=False)[0]
    vehicle_boxes = []

    for r in results.boxes:
        cls = int(r.cls[0])
        if cls in [2, 3, 5, 7]:  # 2:car, 3: motorcycle, 5: bus, 7: truck
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            vehicle_boxes.append((x1, y1, x2, y2))

    if len(vehicle_boxes) < 2:
        return False

    for i in range(len(vehicle_boxes)):
        for j in range(i + 1, len(vehicle_boxes)):
            if iou(vehicle_boxes[i], vehicle_boxes[j]) > 0.3:
                return True

    return False
