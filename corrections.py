import cv2
from pydub import AudioSegment
import fitz

def correct_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img

def correct_audio(path, output_path):
    audio = AudioSegment.from_file(path)
    audio = audio.normalize()
    audio.export(output_path, format="mp3")
    return output_path

def correct_pdf(path):
    doc = fitz.open(path)
    return len(doc)
