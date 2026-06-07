import cv2
from detector import detect_collision
from datetime import datetime
from caller import make_call

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Accident Detection", frame)

        if detect_collision(frame):
            send_sos()
            make_call()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def send_sos():
    message = "🚨 Accident Detected at Laptop - " + datetime.now().strftime("%I:%M %p") + "\n"
    with open("C:\\STUDIES\\libraries\\Accident-detection-ai\\sos_data.txt", "a", encoding="utf-8") as file:
        file.write(message)

    import json
    try:
        with open("C:\\STUDIES\\libraries\\Accident-detection-ai\\sos_data.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append({ "time": datetime.now().strftime("%I:%M %p"), "location": "laptop" })

    with open("C:\\STUDIES\\libraries\\Accident-detection-ai\\sos_data.json", "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
