import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
from pdf2image import convert_from_bytes

from src.cv_engine import CVPreprocessingEngine
from src.ocr_engine import LocalOCREngine
from src.extraction_router import HybridExtractionRouter
from src.data_validator import validate_extraction
from src.database import DocumentDatabase

st.set_page_config(page_title="Enterprise AI Document Engine", layout="wide")
st.title("⚙️ Enterprise Document QA, OCR & Data Extraction System")

# Initialize engines & DB
@st.cache_resource
def load_engines():
    return CVPreprocessingEngine(), LocalOCREngine(), DocumentDatabase()

cv_engine, ocr_engine, db = load_engines()

# Create App Navigation Tabs
tab_extract, tab_analytics = st.tabs(["📄 Document Processing", "📊 System Audit & Analytics"])

# Sidebar Configuration
with st.sidebar:
    st.header("🔑 AI Configuration")
    secret_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
    gemini_key = st.text_input("Gemini API Key (Optional Fallback)", value=secret_key, type="password")
    conf_thresh = st.slider("OCR Confidence Threshold for LLM Escalation", 0.50, 0.95, 0.85)

router = HybridExtractionRouter(gemini_api_key=gemini_key)

with tab_extract:
    uploaded_file = st.file_uploader("Upload Document (Invoice, Receipt, PDF)", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file:
        if uploaded_file.name.lower().endswith(".pdf"):
            images = convert_from_bytes(uploaded_file.read())
            raw_image = images[0]
        else:
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
            st.subheader("2. Deep Learning OCR & Hybrid Routing")
            
            if st.button("🚀 Process Document Pipeline", type="primary"):
                with st.spinner("Running Local OCR Engine..."):
                    ocr_result = ocr_engine.extract_text_and_boxes(processed_np)
                
                st.info(f"📊 **OCR Confidence Score:** {ocr_result['avg_confidence'] * 100:.1f}%")
                
                # Draw Bounding Boxes
                annotated_img = processed_np.copy()
                for item in ocr_result["bounding_boxes"]:
                    box = np.array(item["box"], dtype=np.int32)
                    cv2.polylines(annotated_img, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="OCR Bounding Boxes Detected", use_container_width=True)
                
                extracted_data = router.route_and_extract(ocr_result, raw_image, confidence_threshold=conf_thresh)
                st.session_state["extracted_data"] = extracted_data

            if "extracted_data" in st.session_state:
                extracted_data = st.session_state["extracted_data"]
                
                st.subheader("3. Structured Data Output")
                st.caption(f"**Extraction Method:** {extracted_data.get('extraction_method')}")
                
                is_valid, warnings, _ = validate_extraction(extracted_data)
                if warnings:
                    for w in warnings:
                        st.warning(f"⚠️ {w}")

                st.subheader("4. Human-in-the-Loop Spreadsheet & Verification")
                with st.form("hitl_form"):
                    f_col1, f_col2, f_col3 = st.columns(3)
                    doc_num = f_col1.text_input("Document Number", value=str(extracted_data.get("document_number") or ""))
                    vendor = f_col2.text_input("Vendor Name", value=str(extracted_data.get("vendor_name") or ""))
                    total = f_col3.number_input("Total Amount ($)", value=float(extracted_data.get("total_amount") or 0.0))
                    
                    st.markdown("**Itemized Line Items Table (Interactive Spreadsheet):**")
                    raw_items = extracted_data.get("line_items", [])
                    items_df = pd.DataFrame(raw_items if raw_items else [{"description": "Item 1", "quantity": 1.0, "unit_price": total, "total": total}])
                    
                    # Editable Streamlit Data Table
                    edited_df = st.data_editor(items_df, num_rows="dynamic", use_container_width=True)

                    submitted = st.form_submit_button("💾 Save to DB & Export JSON")
                    if submitted:
                        final_payload = {
                            "document_number": doc_num,
                            "vendor_name": vendor,
                            "total_amount": total,
                            "line_items": edited_df.to_dict(orient="records"),
                            "human_verified": True
                        }
                        
                        # Save transaction record to SQLite Database
                        db.save_record(
                            filename=uploaded_file.name,
                            doc_number=doc_num,
                            vendor=vendor,
                            total=total,
                            confidence=extracted_data.get("confidence_score", 0.0),
                            method=extracted_data.get("extraction_method", "Unknown"),
                            verified=True,
                            payload=final_payload
                        )
                        
                        st.success("Record saved to SQLite database!")
                        st.json(final_payload)
                        st.download_button("📥 Download Verified JSON Payload", data=json.dumps(final_payload, indent=2), file_name="verified_document.json", mime="application/json")

with tab_analytics:
    st.subheader("📊 System Performance & Database Audit Log")
    
    try:
        df_history = db.fetch_all_records()
        if not df_history.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Processed Docs", len(df_history))
            m2.metric("Local OCR Routing Rate", f"{(df_history['extraction_method'].str.contains('Local').mean() * 100):.1f}%")
            m3.metric("Avg OCR Confidence", f"{(df_history['ocr_confidence'].mean() * 100):.1f}%")
            
            st.markdown("### Processed Document Logs")
            st.dataframe(df_history, use_container_width=True)
        else:
            st.info("No records saved in database yet. Process a document to view analytics.")
    except Exception as e:
        st.info("Database initialized. Process your first document to populate metrics.")
