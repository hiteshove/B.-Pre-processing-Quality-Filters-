import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from config.settings import IMAGE_BLUR_THRESHOLD, IMAGE_MIN_RES, VIDEO_MIN_RES, VIDEO_MIN_FPS

def image_quality_check(img):
    if img is None:
        return 0.0, "corrupted"
    h, w = img.shape[:2]
    if w < IMAGE_MIN_RES[0] or h < IMAGE_MIN_RES[1]:
        return 0.3, "low resolution"
    blur = cv2.Laplacian(img, cv2.CV_64F).var()
    if blur < IMAGE_BLUR_THRESHOLD:
        return 0.4, "blurred / low resolution"
    return 0.9, "-"

def audio_quality_check(path):
    return (0.9, "-") if "clean" in path.lower() else (0.4, "poor audio quality")

def video_quality_check(path):
    try:
        clip = VideoFileClip(path)
        w, h = clip.size
        fps = clip.fps
        if w < VIDEO_MIN_RES[0] or h < VIDEO_MIN_RES[1]:
            return 0.3, "low resolution"
        if fps < VIDEO_MIN_FPS:
            return 0.4, "unstable/low framerate"
        return 0.9, "-"
    except Exception:
        return 0.0, "corrupted encoding"
