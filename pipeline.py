
"""
Gen AI Data Ingestion System — Improved
Pre-processing, Quality Filters & Classification
- Saves corrected assets alongside originals
- Uses optional deps where available (Pillow, mutagen, librosa)
- More robust image/video quality checks
- Consistent reason taxonomy & thresholds
- Deterministic, testable design
"""

import os
import io
import sys
import csv
import json
import math
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# Optional imports guarded
try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image, ImageOps, ImageFilter, ExifTags
except Exception:
    Image = ImageOps = ImageFilter = ExifTags = None

# Audio optional libs
try:
    import mutagen
except Exception:
    mutagen = None

try:
    import librosa
except Exception:
    librosa = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingestion")

# ==================== Configuration ====================
class Config:
    """Configuration settings for quality thresholds"""
    # Image thresholds
    IMAGE_MIN_RESOLUTION = (640, 480)  # (w, h)
    IMAGE_BLUR_THRESHOLD = 120.0       # Variance of Laplacian
    IMAGE_NOISE_MAX = 55.0             # stddev threshold for noise proxy
    IMAGE_MIN_QUALITY_SCORE = 0.5

    # Audio thresholds
    AUDIO_MIN_BITRATE = 96             # kbps (fallback when metadata available)
    AUDIO_MIN_RMS = -36.0              # dBFS (approx, when librosa available)
    AUDIO_MIN_QUALITY_SCORE = 0.5

    # Video thresholds
    VIDEO_MIN_RESOLUTION = (640, 480)  # (w, h)
    VIDEO_MIN_FPS = 24
    VIDEO_MIN_QUALITY_SCORE = 0.5

    # Document settings (non-image docs)
    DOCUMENT_MIN_BYTES = 200

    # Output settings
    OUTPUT_DIR = "processed_assets"
    REPORT_FILE = "ingestion_report.csv"

    # Reasons taxonomy
    REASONS = {
        "OK": "-",
        "IMG_LOW_RES": "low resolution",
        "IMG_BLUR": "excessive blur",
        "IMG_NOISE": "excessive noise",
        "CORRUPTED": "corrupted",
        "AUDIO_LOW_BITRATE": "low bitrate",
        "AUDIO_POOR_QUALITY": "poor audio quality",
        "VIDEO_LOW_RES": "low resolution",
        "VIDEO_LOW_FPS": "unstable/low framerate",
        "VIDEO_CORRUPTED": "corrupted encoding",
        "DOC_INSUFFICIENT": "insufficient content",
        "PROCESSING_ERROR": "processing error",
        "QUALITY_BELOW": "quality below threshold",
        "UNSUPPORTED": "unsupported type"
    }

# ==================== Utilities ====================
def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_image_cv(out_path: Path, img_array: np.ndarray) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV not available to write image")
    safe_mkdir(out_path.parent)
    cv2.imwrite(str(out_path), img_array)

def pil_to_cv(img_pil):
    return np.array(img_pil)[:, :, ::-1].copy()

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# ==================== File Ingestion ====================
class FileIngestor:
    """Handles file ingestion and initial cataloging"""
    SUPPORTED_EXTENSIONS = {
        'image': ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'],
        'document': ['pdf', 'docx', 'txt', 'doc', 'rtf', 'odt'],
        'audio': ['mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg'],
        'video': ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'webm']
    }

    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.records = []
        self.asset_counter = 1

    def ingest(self) -> pd.DataFrame:
        logger.info(f"Starting ingestion from: {self.input_dir}")
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_path = Path(root) / file
                ftype = self._determine_file_type(file_path)
                if not ftype:
                    continue
                try:
                    record = {
                        'asset_id': f'{self.asset_counter:03d}',
                        'filename': file,
                        'path': str(file_path),
                        'sha1': file_sha1(file_path),
                        'type': ftype,
                        'quality': None,
                        'status': 'pending',
                        'reason': '-',
                        'subtype': '-',
                        'corrected_path': ''
                    }
                    self.records.append(record)
                    self.asset_counter += 1
                except Exception as e:
                    logger.warning(f"Skipping {file}: {e}")
        logger.info(f"Ingested {len(self.records)} files")
        return pd.DataFrame(self.records)

    def _determine_file_type(self, file_path: Path) -> Optional[str]:
        ext = file_path.suffix.lower().strip('.')
        for ftype, exts in self.SUPPORTED_EXTENSIONS.items():
            if ext in exts:
                return ftype
        return None

# ==================== Corrections Module ====================
class AssetCorrector:
    """Applies automatic corrections to improve asset quality"""

    @staticmethod
    def correct_image(img_path: str) -> Optional[np.ndarray]:
        try:
            # Prefer PIL for EXIF-aware autorotate, then refine in OpenCV
            img_array = None

            if Image is not None:
                with Image.open(img_path) as im:
                    # Autorotate using EXIF if available
                    try:
                        im = ImageOps.exif_transpose(im)
                    except Exception:
                        pass
                    # Mild contrast/brightness and denoise/sharpen
                    im = im.filter(ImageFilter.MedianFilter(size=3))
                    # Convert to OpenCV array for further ops
                    img_array = pil_to_cv(im) if cv2 is not None else np.array(im)
            elif cv2 is not None:
                img_array = cv2.imread(img_path)
                if img_array is None:
                    return None
            else:
                return None

            if cv2 is not None:
                # Auto-contrast & brightness
                lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl,a,b))
                img_array = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                # Mild denoise
                img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 7, 7, 7, 21)
                # Unsharp mask
                gaussian = cv2.GaussianBlur(img_array, (0,0), 1.0)
                img_array = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)

            return img_array
        except Exception as e:
            logger.error(f"Error correcting image {img_path}: {e}")
            return None

    @staticmethod
    def correct_document(doc_path: str) -> Dict[str, Any]:
        corrections = {'applied': False, 'details': []}
        try:
            # If it's a scanned image masquerading as document, reuse image pipeline
            lower = doc_path.lower()
            if lower.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')):
                img = AssetCorrector.correct_image(doc_path)
                if img is not None:
                    corrections['applied'] = True
                    corrections['details'].append('deskew/denoise/sharpen')
                    corrections['corrected_image'] = img
        except Exception as e:
            logger.error(f"Error correcting document {doc_path}: {e}")
        return corrections

    @staticmethod
    def correct_audio(audio_path: str) -> Dict[str, Any]:
        corrections = {'applied': False, 'details': []}
        try:
            # If librosa available, normalize peak level
            if librosa is not None:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
                peak = np.max(np.abs(y)) + 1e-9
                y = y / peak * 0.95
                corrections['applied'] = True
                corrections['details'].append('normalized peak to -0.5 dBFS (approx)')
                corrections['audio_array'] = y
                corrections['sr'] = sr
            else:
                corrections['applied'] = True
                corrections['details'].append('normalization (placeholder)')
        except Exception as e:
            logger.error(f"Error correcting audio {audio_path}: {e}")
        return corrections

    @staticmethod
    def correct_video(video_path: str) -> Dict[str, Any]:
        corrections = {'applied': False, 'details': []}
        try:
            # Real stabilization/denoise would require heavy deps; leave as placeholder
            corrections['applied'] = True
            corrections['details'].append('basic validation, placeholder corrections')
        except Exception as e:
            logger.error(f"Error correcting video {video_path}: {e}")
        return corrections

# ==================== Quality Filters ====================
class QualityFilter:
    """Applies quality filters to determine asset acceptability"""

    @staticmethod
    def check_image_quality(img_path: str) -> Tuple[float, str]:
        try:
            if cv2 is None:
                return 0.0, Config.REASONS["PROCESSING_ERROR"]
            img = cv2.imread(img_path)
            if img is None:
                return 0.0, Config.REASONS["CORRUPTED"]

            h, w = img.shape[:2]
            if w < Config.IMAGE_MIN_RESOLUTION[0] or h < Config.IMAGE_MIN_RESOLUTION[1]:
                return 0.35, Config.REASONS["IMG_LOW_RES"]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Noise proxy (residual after Gaussian blur)
            residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0)
            noise_level = float(np.std(residual))

            if blur_score < Config.IMAGE_BLUR_THRESHOLD:
                return 0.42, Config.REASONS["IMG_BLUR"]
            if noise_level > Config.IMAGE_NOISE_MAX:
                return 0.45, Config.REASONS["IMG_NOISE"]

            # Composite score
            res_score = min(w / 1920, h / 1080, 1.0)
            blur_norm = min(blur_score / 600, 1.0)
            noise_norm = max(0, 1 - noise_level / 100)
            quality = (0.35 * res_score + 0.5 * blur_norm + 0.15 * noise_norm)

            reason = Config.REASONS["OK"] if quality >= Config.IMAGE_MIN_QUALITY_SCORE else Config.REASONS["QUALITY_BELOW"]
            return round(float(quality), 2), reason
        except Exception as e:
            logger.error(f"Error checking image quality for {img_path}: {e}")
            return 0.0, Config.REASONS["PROCESSING_ERROR"]

    @staticmethod
    def _mutagen_bitrate_kbps(audio_path: str) -> Optional[int]:
        if mutagen is None:
            return None
        try:
            f = mutagen.File(audio_path)
            if f is None:
                return None
            # Many formats expose .info.bitrate in bps
            if hasattr(f, "info") and hasattr(f.info, "bitrate") and f.info.bitrate:
                return int(f.info.bitrate // 1000)
        except Exception:
            return None
        return None

    @staticmethod
    def _librosa_rms_dbfs(audio_path: str) -> Optional[float]:
        if librosa is None:
            return None
        try:
            y, _ = librosa.load(audio_path, sr=None, mono=True)
            rms = np.sqrt(np.mean(np.square(y))) + 1e-12
            dbfs = 20 * math.log10(rms)
            return float(dbfs)
        except Exception:
            return None

    @staticmethod
    def check_audio_quality(audio_path: str) -> Tuple[float, str]:
        try:
            # Try bitrate via mutagen
            bitrate = QualityFilter._mutagen_bitrate_kbps(audio_path)
            loudness_dbfs = QualityFilter._librosa_rms_dbfs(audio_path)

            reasons = []
            score_components = []

            if bitrate is not None:
                score_components.append(min(bitrate / 256.0, 1.0))  # normalize 0..1
                if bitrate < Config.AUDIO_MIN_BITRATE:
                    reasons.append(Config.REASONS["AUDIO_LOW_BITRATE"])
            else:
                # Unknown bitrate → neutral component
                score_components.append(0.6)

            if loudness_dbfs is not None:
                # Expect typical RMS around -30..-18 dBFS for decent material
                loud_norm = np.clip((loudness_dbfs + 60) / 40.0, 0.0, 1.0)  # map -60..-20 -> 0..1
                score_components.append(loud_norm)
            else:
                score_components.append(0.6)

            quality = float(np.mean(score_components))
            reason = Config.REASONS["OK"] if (quality >= Config.AUDIO_MIN_QUALITY_SCORE and not reasons) else (
                     reasons[0] if reasons else Config.REASONS["AUDIO_POOR_QUALITY"] if quality < Config.AUDIO_MIN_QUALITY_SCORE else Config.REASONS["OK"])
            return round(quality, 2), reason
        except Exception as e:
            logger.error(f"Error checking audio quality for {audio_path}: {e}")
            return 0.0, Config.REASONS["PROCESSING_ERROR"]

    @staticmethod
    def check_video_quality(video_path: str) -> Tuple[float, str]:
        try:
            if cv2 is None:
                return 0.0, Config.REASONS["PROCESSING_ERROR"]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0, Config.REASONS["VIDEO_CORRUPTED"]
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()

            if w < Config.VIDEO_MIN_RESOLUTION[0] or h < Config.VIDEO_MIN_RESOLUTION[1]:
                return 0.35, Config.REASONS["VIDEO_LOW_RES"]
            if fps < Config.VIDEO_MIN_FPS:
                return 0.45, Config.REASONS["VIDEO_LOW_FPS"]

            # Size-based proxy and fps normalization
            res_score = min(w / 1920, h / 1080, 1.0)
            fps_norm = min(fps / 30.0, 1.0)
            quality = 0.6 * res_score + 0.4 * fps_norm
            reason = Config.REASONS["OK"] if quality >= Config.VIDEO_MIN_QUALITY_SCORE else Config.REASONS["QUALITY_BELOW"]
            return round(quality, 2), reason
        except Exception as e:
            logger.error(f"Error checking video quality for {video_path}: {e}")
            return 0.0, Config.REASONS["PROCESSING_ERROR"]

    @staticmethod
    def check_document_quality(doc_path: str) -> Tuple[float, str]:
        try:
            lower = doc_path.lower()
            if lower.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')):
                return QualityFilter.check_image_quality(doc_path)
            size = os.path.getsize(doc_path)
            if size < Config.DOCUMENT_MIN_BYTES:
                return 0.2, Config.REASONS["DOC_INSUFFICIENT"]
            return 0.9, Config.REASONS["OK"]
        except Exception as e:
            logger.error(f"Error checking document quality for {doc_path}: {e}")
            return 0.0, Config.REASONS["PROCESSING_ERROR"]

# ==================== Classification ====================
class AssetClassifier:
    """Classifies assets into subcategories via lightweight heuristics"""

    @staticmethod
    def classify_document(doc_path: str) -> str:
        # Placeholder: without OCR, assume printed
        return "printed"

    @staticmethod
    def classify_image(img_path: str) -> str:
        if cv2 is None:
            return "unknown"
        try:
            img = cv2.imread(img_path)
            if img is None:
                return "unknown"
            # Edge density heuristic: scans have sharp edges; photos vary
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            density = float(np.mean(edges > 0))
            return "scan" if density > 0.12 else "photograph"
        except Exception:
            return "unknown"

    @staticmethod
    def classify_audio(audio_path: str) -> str:
        # Simple heuristic: if noisy name or low bitrate
        if "clean" in audio_path.lower():
            return "clean"
        br = QualityFilter._mutagen_bitrate_kbps(audio_path)
        return "noisy" if (br is not None and br < 128) else "clean"

    @staticmethod
    def classify_video(video_path: str) -> str:
        return "valid"

# ==================== Main Pipeline ====================
class DataIngestionPipeline:
    """Main pipeline orchestrating the entire process"""

    def __init__(self, input_dir: str, output_dir: str = None):
        self.input_dir = input_dir
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.df: Optional[pd.DataFrame] = None
        safe_mkdir(Path(self.output_dir))

    def run(self) -> pd.DataFrame:
        logger.info("Starting Data Ingestion Pipeline (improved)")
        ingestor = FileIngestor(self.input_dir)
        self.df = ingestor.ingest()

        for idx, row in self.df.iterrows():
            self._process_asset(idx, row)

        self._generate_report()
        logger.info("Pipeline completed successfully")
        return self.df

    def _save_corrected(self, src_path: str, corrected: Any, suffix: str) -> str:
        try:
            src = Path(src_path)
            dest_dir = Path(self.output_dir) / src.parent.name
            safe_mkdir(dest_dir)
            dest = dest_dir / f"{src.stem}.corrected{suffix}"
            if isinstance(corrected, np.ndarray) and cv2 is not None:
                cv2.imwrite(str(dest), corrected)
            elif isinstance(corrected, bytes):
                with open(dest, "wb") as f:
                    f.write(corrected)
            # For audio arrays or others, skip actual write for now
            return str(dest)
        except Exception as e:
            logger.warning(f"Could not save corrected asset for {src_path}: {e}")
            return ""

    def _process_asset(self, idx: int, row: pd.Series):
        asset_type = row['type']
        asset_path = row['path']

        logger.info(f"Processing {row['asset_id']}: {row['filename']}")

        quality = 0.0
        reason = Config.REASONS["UNSUPPORTED"]
        subtype = "unknown"
        corrected_path = ""

        try:
            if asset_type == 'image':
                corr = AssetCorrector.correct_image(asset_path)
                if corr is not None:
                    corrected_path = self._save_corrected(asset_path, corr, Path(asset_path).suffix)
                quality, reason = QualityFilter.check_image_quality(asset_path)
                subtype = AssetClassifier.classify_image(asset_path)
                threshold = Config.IMAGE_MIN_QUALITY_SCORE

            elif asset_type == 'document':
                dcorr = AssetCorrector.correct_document(asset_path)
                if dcorr.get('corrected_image') is not None:
                    corrected_path = self._save_corrected(asset_path, dcorr['corrected_image'], Path(asset_path).suffix)
                quality, reason = QualityFilter.check_document_quality(asset_path)
                subtype = AssetClassifier.classify_document(asset_path)
                threshold = 0.5

            elif asset_type == 'audio':
                acorr = AssetCorrector.correct_audio(asset_path)
                # (Optional: write out normalized audio if implemented)
                quality, reason = QualityFilter.check_audio_quality(asset_path)
                subtype = AssetClassifier.classify_audio(asset_path)
                threshold = Config.AUDIO_MIN_QUALITY_SCORE

            elif asset_type == 'video':
                vcorr = AssetCorrector.correct_video(asset_path)
                quality, reason = QualityFilter.check_video_quality(asset_path)
                subtype = AssetClassifier.classify_video(asset_path)
                threshold = Config.VIDEO_MIN_QUALITY_SCORE

            else:
                threshold = 0.5

            status = 'accepted' if quality >= threshold and reason in (Config.REASONS["OK"], Config.REASONS["QUALITY_BELOW"]) else ('accepted' if quality >= threshold and asset_type != 'image' else 'rejected')
        except Exception as e:
            logger.error(f"Error processing asset {asset_path}: {e}")
            status = 'rejected'
            reason = Config.REASONS["PROCESSING_ERROR"]

        # Update df
        self.df.at[idx, 'quality'] = round(float(quality), 2) if isinstance(quality, (int, float)) else quality
        self.df.at[idx, 'reason'] = reason
        self.df.at[idx, 'subtype'] = subtype
        self.df.at[idx, 'status'] = status
        self.df.at[idx, 'corrected_path'] = corrected_path

    def _generate_report(self):
        report_path = Path(self.output_dir) / Config.REPORT_FILE
        report_df = self.df[['asset_id', 'filename', 'type', 'quality', 'status', 'reason', 'subtype', 'corrected_path']]
        report_df.to_csv(report_path, index=False)
        logger.info(f"Report saved to: {report_path}")
        print("\\n" + "="*60)
        print("INGESTION SUMMARY (IMPROVED)")
        print("="*60)
        print(f"Total Assets Processed: {len(self.df)}")
        print(f"Accepted: {len(self.df[self.df['status'] == 'accepted'])}")
        print(f"Rejected: {len(self.df[self.df['status'] == 'rejected'])}")
        print("\\nBy Type:")
        print(self.df.groupby('type')['status'].value_counts().unstack(fill_value=0))
        print("\\nSample Results:")
        print(report_df.head(10).to_string())

if __name__ == "__main__":  # pragma: no cover
    # CLI-like quick start (edit paths as needed)
    input_directory = "./input_assets"
    output_directory = "./processed_assets"
    pipeline = DataIngestionPipeline(input_directory, output_directory)
    try:
        pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
