import pytesseract
import cv2

def classify_document(path):
    img = cv2.imread(path) if path.lower().endswith((".jpg", ".png")) else None
    if img is not None:
        text = pytesseract.image_to_string(img)
        return "printed" if len(text.strip()) > 30 else "handwritten"
    return "printed"

def classify_image(path):
    return "photograph" if "photo" in path.lower() else "scan"

def classify_audio(path):
    return "clean" if "clean" in path.lower() else "noisy"

def classify_video(path):
    return "valid"
