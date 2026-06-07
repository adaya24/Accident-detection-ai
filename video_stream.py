import cv2

def load_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video stream or file")
    return cap
