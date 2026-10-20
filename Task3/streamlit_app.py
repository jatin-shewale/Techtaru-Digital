"""Streamlit interface for the Task3 resume and invoice extraction bot."""
from __future__ import annotations

import hashlib
import json
import logging
from io import BytesIO

import streamlit as st
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pypdf import PdfReader

from extraction_core import (
    MAX_INPUT_CHARS,
    SCHEMAS,
    ExtractionError,
    Invoice,
    Resume,
    extract,
    validate,
)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("task3.streamlit")

st.set_page_config(page_title="Data Extraction Bot", page_icon="🧾", layout="wide")
st.title("🧾 Data Extraction Bot")
st.caption("Turn resume or invoice text into structured, validated data with Groq.")

with st.sidebar:
    st.header("Extraction settings")
    doc_type = st.radio(
        "What are you extracting?",
        options=["resume", "invoice"],
        format_func=lambda value: value.title(),
    )
    st.divider()
    st.markdown("**Input options**")
    uploaded = st.file_uploader("Load a PDF or plain text file", type=["pdf", "txt"])
    st.caption("You can also paste text directly. Scanned PDFs need OCR before upload.")

st.session_state.setdefault("document_text", "")
st.session_state.setdefault("last_upload_digest", None)

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_digest = hashlib.sha256(file_bytes).hexdigest()
    if file_digest != st.session_state.last_upload_digest:
        st.session_state.pop("upload_status", None)
        try:
            if uploaded.name.lower().endswith(".pdf"):
                reader = PdfReader(BytesIO(file_bytes))
                if reader.is_encrypted:
                    raise ValueError("This PDF is password-protected. Upload an unlocked PDF.")
                pages = []
                for page_number, page in enumerate(reader.pages, start=1):
                    page_text = " ".join((page.extract_text() or "").split())
                    if page_text:
                        pages.append(f"[Page {page_number}]\n{page_text}")
                if not pages:
                    raise ValueError("No extractable text found in this PDF. It may be scanned; run OCR first.")
                st.session_state.document_text = "\n\n".join(pages)
                st.session_state.upload_status = f"Loaded PDF text from {len(pages)} page(s)."
            else:
                st.session_state.document_text = file_bytes.decode("utf-8-sig")
                st.session_state.upload_status = "Loaded text file."
            st.session_state.last_upload_digest = file_digest
        except UnicodeDecodeError:
            st.sidebar.error("Could not read this file as UTF-8 text. Save it as a UTF-8 .txt file and try again.")
        except ValueError as exc:
            st.sidebar.error(str(exc))
        except Exception:
            log.exception("Could not read uploaded document")
            st.sidebar.error("Could not read this file. Check that it is a valid PDF or UTF-8 text file.")

if st.session_state.get("upload_status"):
    st.sidebar.success(st.session_state.upload_status)

st.subheader(f"{doc_type.title()} text")
st.write("Paste document text below, or load a PDF or `.txt` file from the sidebar.")

with st.form("extract_form", clear_on_submit=False):
    input_text = st.text_area(
        "Document text",
        key="document_text",
        height=300,
        placeholder=(
            "Paste resume text here..." if doc_type == "resume"
            else "Paste invoice text here..."
        ),
        label_visibility="collapsed",
    )
    left, right = st.columns([1, 5])
    with left:
        submitted = st.form_submit_button("Extract fields", type="primary", use_container_width=True)
    with right:
        st.caption(f"{len(input_text):,} characters · maximum processed: {MAX_INPUT_CHARS:,}")

if len(input_text) > MAX_INPUT_CHARS:
    st.info(f"The extractor will process the first {MAX_INPUT_CHARS:,} characters.")

if submitted:
    st.session_state.pop("extraction_fingerprint", None)
    if not input_text.strip():
        st.warning("Add some document text before starting extraction.")
    else:
        try:
            with st.spinner("Extracting and validating fields…"):
                parsed = extract(input_text, doc_type)
                warnings = validate(parsed)
            st.session_state.extraction_result = parsed.model_dump(mode="json")
            st.session_state.extraction_warnings = warnings
            st.session_state.extraction_type = doc_type
            st.session_state.extraction_fingerprint = hashlib.sha256(
                f"{doc_type}\0{input_text}".encode("utf-8")
            ).hexdigest()
        except RateLimitError:
            st.error("Groq is rate-limiting requests right now. Wait a moment, then try again.")
        except (APIConnectionError, APITimeoutError):
            st.error("Could not reach the Groq API. Check your connection and try again.")
        except APIStatusError as exc:
            log.exception("Groq API returned an error")
            if exc.status_code == 413:
                st.error(
                    "Groq rejected this request as too large for the current token limit. "
                    "Use a shorter document or split it into smaller sections and extract each section separately."
                )
            else:
                st.error(f"Groq API error ({exc.status_code}). Check your API key and selected model.")
        except RuntimeError as exc:
            st.error(str(exc))
        except (ValueError, ExtractionError) as exc:
            st.error(str(exc))
        except Exception:
            log.exception("Unexpected extraction failure")
            st.error("Extraction failed unexpectedly. See the terminal output for details.")

current_fingerprint = hashlib.sha256(f"{doc_type}\0{input_text}".encode("utf-8")).hexdigest()
if (
    st.session_state.get("extraction_fingerprint") == current_fingerprint
    and st.session_state.get("extraction_type") == doc_type
):
    result = st.session_state.extraction_result
    warnings = st.session_state.extraction_warnings
    st.divider()
    st.subheader("Extraction result")

    if warnings:
        st.warning("Review these validation warnings:")
        for warning in warnings:
            st.markdown(f"- {warning}")
    else:
        st.success("Extraction complete. No validation warnings.")

    model = SCHEMAS[doc_type]
    if model is Resume:
        top = st.columns(3)
        top[0].metric("Name", result.get("full_name") or "Not found")
        top[1].metric("Email", result.get("email") or "Not found")
        top[2].metric("Phone", result.get("phone") or "Not found")
        st.markdown("**Skills**")
        st.write(", ".join(result.get("skills") or []) or "No skills found")
        details = st.columns(2)
        with details[0]:
            st.markdown("**Experience**")
            st.dataframe(result.get("experience") or [], use_container_width=True, hide_index=True)
        with details[1]:
            st.markdown("**Education**")
            st.dataframe(result.get("education") or [], use_container_width=True, hide_index=True)
    elif model is Invoice:
        first = st.columns(3)
        first[0].metric("Invoice number", result.get("invoice_number") or "Not found")
        first[1].metric("Vendor", result.get("vendor") or "Not found")
        first[2].metric("Customer", result.get("customer") or "Not found")
        totals = st.columns(4)
        for column, label, field in zip(
            totals,
            ["Subtotal", "Tax", "Total", "Currency"],
            ["subtotal", "tax", "total", "currency"],
        ):
            value = result.get(field)
            column.metric(label, value if value is not None else "Not found")
        st.markdown("**Line items**")
        st.dataframe(result.get("line_items") or [], use_container_width=True, hide_index=True)

    with st.expander("View complete JSON"):
        st.json(result)
    st.download_button(
        "Download JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name=f"{doc_type}_extraction.json",
        mime="application/json",
    )

with st.expander("What gets extracted?"):
    if doc_type == "resume":
        st.markdown("Name, contact details, skills, education, and work experience.")
    else:
        st.markdown("Invoice number, vendor and customer, dates, currency, line items, subtotal, tax, and total.")
    st.caption("Only fields supported by the source text are returned. Missing values remain empty or null.")
