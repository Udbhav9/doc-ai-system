import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json

from src.cv_engine import CVPreprocessingEngine
from src.ocr_engine import LocalOCREngine
from src.extraction_router import HybridExtractionRouter
from src.data_validator import validate_extraction

st.set_page_config(page_title="AI Document Engine", layout="wide")
st.title("⚙️ Intelligent Document QA, OCR & Data Extraction System")

# Initialize engines
@st.cache_resource
def load_engines():
    return CVPreprocessingEngine(), LocalOCREngine()

cv_engine, ocr_engine = load_engines()

# Sidebar Configuration
with st.sidebar:
    st.header("🔑 AI Configuration")
    # Tries retrieving key from Streamlit Cloud Secrets first, falls back to text input
    secret_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
    gemini_key = st.text_input("Gemini API Key (Optional Fallback)", value=secret_key, type="password")
    conf_thresh = st.slider("OCR Confidence Threshold for LLM Escalation", 0.50, 0.95, 0.85)

router = HybridExtractionRouter(gemini_api_key=gemini_key)

uploaded_file = st.file_uploader("Upload Document (Invoice, Receipt)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Computer Vision Quality Check & Deskew")
        processed_np, cv_report = cv_engine.preprocess(raw_image)
        
        if cv_report["status"] == "PASS":
            st.success(f"✅ Quality Passed (Blur Score: {cv_report['blur_score']})")
        else:
            st.error(f"⚠️ Low Quality Detected (Blur Score: {cv_report['blur_score']})")
            
        st.image(cv2.cvtColor(processed_np, cv2.COLOR_BGR2RGB), caption="Deskewed Image", use_container_width=True)

    with col2:
        st.subheader("2. Local Deep Learning OCR & Hybrid Routing")
        
        if st.button("🚀 Process Document Pipeline", type="primary"):
            with st.spinner("Running PaddleOCR Engine (DBNet + SVTR)..."):
                ocr_result = ocr_engine.extract_text_and_boxes(processed_np)
            
            st.info(f"📊 **OCR Confidence Score:** {ocr_result['avg_confidence'] * 100:.1f}%")
            
            # Draw Bounding Boxes
            annotated_img = processed_np.copy()
            for item in ocr_result["bounding_boxes"]:
                box = np.array(item["box"], dtype=np.int32)
                cv2.polylines(annotated_img, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="OCR Bounding Boxes Detected", use_container_width=True)
            
            # Route and extract data
            extracted_data = router.route_and_extract(ocr_result, raw_image, confidence_threshold=conf_thresh)
            st.session_state["extracted_data"] = extracted_data

        # Human-in-the-Loop Section
        if "extracted_data" in st.session_state:
            extracted_data = st.session_state["extracted_data"]
            
            st.subheader("3. Structured Data Output")
            st.caption(f"**Extraction Method:** {extracted_data.get('extraction_method')}")
            
            # Run Data Audit Rules
            is_valid, warnings, _ = validate_extraction(extracted_data)
            if warnings:
                for w in warnings:
                    st.warning(f"⚠️ {w}")

            st.subheader("4. Human-in-the-Loop Review & Export")
            with st.form("hitl_form"):
                doc_num = st.text_input("Document Number", value=str(extracted_data.get("document_number") or ""))
                total = st.number_input("Total Amount ($)", value=float(extracted_data.get("total_amount") or 0.0))
                
                submitted = st.form_submit_button("Confirm & Generate Downloadable JSON")
                if submitted:
                    final_payload = {
                        "document_number": doc_num, 
                        "total_amount": total, 
                        "human_verified": True
                    }
                    st.success("Data successfully verified!")
                    st.json(final_payload)
                    
                    # Direct browser download (ideal for cloud hosting)
                    st.download_button(
                        label="📥 Download Verified JSON Payload",
                        data=json.dumps(final_payload, indent=2),
                        file_name="verified_document.json",
                        mime="application/json"
                    )
