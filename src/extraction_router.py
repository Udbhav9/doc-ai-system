import re
import json
import os
from PIL import Image
from google import genai
from google.genai import types

class HybridExtractionRouter:
    """Routes document processing through Regex/Rule-based parsing or LLM vision fallback."""

    def __init__(self, gemini_api_key: str = None):
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def parse_with_rules(self, text: str) -> dict:
        """Rule-Based / Regex NER Extraction Engine."""
        invoice_num = re.search(r'(?i)(?:invoice|inv|bill)\s*(?:#|no|num)?[:\s]*([a-zA-Z0-9-]+)', text)
        total_amt = re.search(r'(?i)(?:total|amount due|grand total)\s*[:\s]*\$?\s*([\d,]+\.\d{2})', text)
        date_match = re.search(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b', text)

        return {
            "document_number": invoice_num.group(1) if invoice_num else None,
            "total_amount": float(total_amt.group(1).replace(',', '')) if total_amt else 0.0,
            "date": date_match.group(0) if date_match else None,
            "extraction_method": "Local OCR + Regex (Fast & Free)"
        }

    def parse_with_vision_llm(self, pil_image: Image.Image) -> dict:
        """Fallback to Vision LLM when local OCR confidence is low or document is complex."""
        prompt = """
        Extract data from this document image in JSON:
        {
            "document_number": "string or null",
            "vendor_name": "string or null",
            "date": "YYYY-MM-DD or null",
            "subtotal": 0.0,
            "tax_amount": 0.0,
            "total_amount": 0.0
        }
        Return ONLY valid raw JSON without markdown headers.
        """
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text)
        data["extraction_method"] = "Multimodal Vision-LLM (Fallback)"
        return data

    def route_and_extract(self, ocr_result: dict, pil_image: Image.Image, confidence_threshold: float = 0.85) -> dict:
        """Intelligent Router decision logic."""
        avg_conf = ocr_result["avg_confidence"]
        
        # High confidence -> Local Regex NER
        if avg_conf >= confidence_threshold:
            extracted_data = self.parse_with_rules(ocr_result["full_text"])
            extracted_data["confidence_score"] = avg_conf
            return extracted_data
        
        # Low confidence -> Fallback to Vision LLM
        if self.api_key:
            extracted_data = self.parse_with_vision_llm(pil_image)
            extracted_data["confidence_score"] = avg_conf
            return extracted_data
        else:
            extracted_data = self.parse_with_rules(ocr_result["full_text"])
            extracted_data["confidence_score"] = avg_conf
            return extracted_data
