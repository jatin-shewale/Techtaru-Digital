"""app.py - Streamlit UI for Chat-with-your-PDF.  Run:  streamlit run app.py"""
import hashlib
import tempfile

import streamlit as st
from groq import RateLimitError

from rag_core import PDFChatbot

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Answers come strictly from the uploaded document.")

try:
    if "bot" not in st.session_state:
        st.session_state.bot = PDFChatbot()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

upload = st.sidebar.file_uploader("Upload a PDF", type="pdf")
if upload is not None:
    digest = hashlib.md5(upload.getvalue()).hexdigest()
    if st.session_state.get("digest") != digest:          # only re-index when the file changes
        with st.spinner("Reading and indexing..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(upload.getvalue())
            try:
                n = st.session_state.bot.ingest(f.name)
                st.session_state.digest, st.session_state.history = digest, []
                st.sidebar.success(f"Indexed {n} chunks")
            except ValueError as e:
                st.sidebar.error(str(e))
            except RateLimitError:
                st.sidebar.error("Rate limited by the API. Try again in a moment.")

for role, content in st.session_state.get("history", []):
    st.chat_message(role).write(content)

if q := st.chat_input("Ask something about the document"):
    st.chat_message("user").write(q)
    try:
        ans = st.session_state.bot.ask(q)
    except RateLimitError:
        st.warning("The API is rate-limiting us. Please retry in a few seconds.")
        st.stop()
    with st.chat_message("assistant"):
        st.write(ans.text)
        if ans.sources:
            with st.expander("Sources"):
                for s in ans.sources:
                    st.markdown(f"**Page {s.page}** (similarity {s.similarity})\n\n{s.text}")
    st.session_state.setdefault("history", []).extend([("user", q), ("assistant", ans.text)])
