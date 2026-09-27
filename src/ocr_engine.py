import numpy as np
from paddleocr import PaddleOCR

class LocalOCREngine:
    def __init__(self):
        # Initialize PaddleOCR (Runs on CPU/GPU locally)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def extract_text_and_boxes(self, img_np: np.ndarray) -> dict:
        """Executes explicit OCR detection and recognition."""
        result = self.ocr.ocr(img_np, cls=True)
        
        extracted_lines = []
        bounding_boxes = []
        confidence_scores = []

        if result and result[0]:
            for line in result[0]:
                box = line[0]  # Bounding box coordinates [[x1,y1], [x2,y2], ...]
                text, conf = line[1][0], line[1][1]  # Recognized text & probability
                
                extracted_lines.append(text)
                bounding_boxes.append({"text": text, "box": box, "confidence": round(float(conf), 2)})
                confidence_scores.append(conf)

        avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        full_text = "\n".join(extracted_lines)

        return {
            "full_text": full_text,
            "avg_confidence": round(avg_confidence, 2),
            "bounding_boxes": bounding_boxes
        }
