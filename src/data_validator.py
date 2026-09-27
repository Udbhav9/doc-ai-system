from pydantic import BaseModel, Field
from typing import Optional, List

class ExtractedDocumentSchema(BaseModel):
    document_number: Optional[str] = None
    date: Optional[str] = None
    vendor_name: Optional[str] = None
    subtotal: float = Field(default=0.0)
    tax_amount: float = Field(default=0.0)
    total_amount: float = Field(default=0.0)
    confidence_score: float = Field(default=1.0)

def validate_extraction(data: dict) -> tuple[bool, List[str], ExtractedDocumentSchema]:
    """Validates raw dict using Pydantic and applies mathematical audit rules."""
    warnings = []
    
    try:
        validated_obj = ExtractedDocumentSchema(**data)
    except Exception as e:
        return False, [f"Schema structural error: {str(e)}"], None

    # Audit Rule: Subtotal + Tax = Total
    if validated_obj.subtotal > 0 and validated_obj.tax_amount > 0:
        expected_total = round(validated_obj.subtotal + validated_obj.tax_amount, 2)
        actual_total = round(validated_obj.total_amount, 2)

        if abs(expected_total - actual_total) > 0.05:
            warnings.append(f"Math Failure: Subtotal ({validated_obj.subtotal}) + Tax ({validated_obj.tax_amount}) = {expected_total}, but Total is {actual_total}")

    if not validated_obj.document_number:
        warnings.append("Missing Document Number.")

    is_valid = len(warnings) == 0
    return is_valid, warnings, validated_obj
