import numpy as np
import easyocr

class LocalOCREngine:
    def __init__(self):
        # Initialize EasyOCR for English running on CPU
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text_and_boxes(self, img_np: np.ndarray) -> dict:
        """Executes explicit OCR detection and recognition using PyTorch/EasyOCR."""
        # easyocr expects RGB image or image path
        results = self.reader.readtext(img_np)
        
        extracted_lines = []
        bounding_boxes = []
        confidence_scores = []

        for bbox, text, conf in results:
            # bbox is a list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            box = [[int(pt[0]), int(pt[1])] for pt in bbox]
            
            extracted_lines.append(text)
            bounding_boxes.append({
                "text": text, 
                "box": box, 
                "confidence": round(float(conf), 2)
            })
            confidence_scores.append(float(conf))

        avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        full_text = "\n".join(extracted_lines)

        return {
            "full_text": full_text,
            "avg_confidence": round(avg_confidence, 2),
            "bounding_boxes": bounding_boxes
        }
