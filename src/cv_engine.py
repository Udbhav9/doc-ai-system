import cv2
import numpy as np
from PIL import Image

class CVPreprocessingEngine:
    """Handles low-level Computer Vision preprocessing before feeding to AI models."""
    
    @staticmethod
    def detect_blur(gray_img: np.ndarray, threshold: float = 100.0) -> tuple[float, bool]:
        """Calculates Laplacian Variance to measure image sharpness/blur."""
        variance = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        return round(float(variance), 2), bool(variance < threshold)

    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """Detects text angle via Hough Line Transform and rotates image straight."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                if abs(angle) < 45:  # Consider horizontal line tilts
                    angles.append(angle)
        
        if angles:
            median_angle = float(np.median(angles))
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image

    def preprocess(self, pil_image: Image.Image) -> tuple[np.ndarray, dict]:
        """Full CV Pipeline: Convert -> Quality Check -> Deskew -> Return clean image."""
        img_np = np.array(pil_image.convert("RGB"))[:, :, ::-1].copy() # Convert PIL to BGR
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        blur_score, is_blurred = self.detect_blur(gray)
        deskewed_np = self.deskew_image(img_np)
        
        quality_report = {
            "blur_score": blur_score,
            "is_blurred": is_blurred,
            "resolution": f"{img_np.shape[1]}x{img_np.shape[0]}",
            "status": "PASS" if not is_blurred else "FAIL"
        }
        
        return deskewed_np, quality_report
