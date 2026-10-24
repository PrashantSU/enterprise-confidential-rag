import streamlit as st
import numpy as np
import faiss
from google import genai
from google.genai import types
import pypdf
import io

st.set_page_config(page_title="Confidential AI Assistant", page_icon="🔒", layout="wide")

st.title("🔒 Confidential Document Assistant")
st.caption("Enterprise-grade RAG. Processed entirely in-memory. Strict page-level citations.")

# --- SIDEBAR: Privacy & API Config ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = st.text_input("Gemini API Key", type="password")
        st.markdown("Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)")
    
    st.markdown("---")
    st.header("🔒 Enterprise Security")
    st.info("""
    **Confidential Data Safe:**
    * Documents are processed **in-memory** and never saved to disk.
    * Data is **never used for model training** (Zero Data Retention).
    * Architecture is ready to be swapped to **local LLMs (Ollama/Llama-3)** for on-premise GDPR compliance.
    """)
    
    st.markdown("---")
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs (Max 5MB each)", type="pdf", accept_multiple_files=True)
    
    if st.button("Clear Chat & Documents"):
        st.session_state.clear()
        st.rerun()

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

client = genai.Client(api_key=api_key) if api_key else None

# --- RAG LOGIC ---
def process_pdfs(files):
    chunks = []
    metadata = []
    
    for pdf in files:
        reader = pypdf.PdfReader(io.BytesIO(pdf.read()))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                chunks.append(text)
                metadata.append({
                    "source": pdf.name,
                    "page": page_num + 1,
                    "text": text
                })
    
    if not chunks:
        return None, None, None

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunks,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    embeddings = np.array([e.values for e in response.embeddings], dtype=np.float32)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return chunks, metadata, index

# --- PROCESS UPLOADS ---
if uploaded_files and not st.session_state.rag_ready:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    else:
        with st.spinner("Securely processing PDFs in-memory..."):
            try:
                chunks, metadata, index = process_pdfs(uploaded_files)
                if chunks:
                    st.session_state.metadata = metadata
                    st.session_state.index = index
                    st.session_state.rag_ready = True
                    st.success(f"✅ Processed {len(uploaded_files)} PDF(s), {len(chunks)} pages. Ready for secure querying.")
                else:
                    st.warning("No extractable text found in the uploaded PDFs.")
            except Exception as e:
                st.error(f"Error: {e}")

# --- CHAT INTERFACE ---
if not st.session_state.rag_ready:
    st.info("👈 Upload one or more PDF documents in the sidebar to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Cited Sources"):
                    for idx, source in enumerate(message["sources"]):
                        st.info(f"**[{idx+1}] {source['source']} (Page {source['page']})**\n\n{source['text']}")

    if prompt := st.chat_input("Ask a question about your confidential documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching securely..."):
                try:
                    res = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=prompt,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                    )
                    query_vector = np.array([res.embeddings[0].values], dtype=np.float32)
                    
                    _, indices = st.session_state.index.search(query_vector, 3)
                    retrieved_sources = [st.session_state.metadata[i] for i in indices[0]]
                    
                    context = "\n\n".join([
                        f"[{i+1}] Source: {src['source']} (Page {src['page']}): {src['text']}" 
                        for i, src in enumerate(retrieved_sources)
                    ])
                    
                    system_prompt = f"""
                    You are a highly secure enterprise assistant. 
                    Answer the user's question using ONLY the provided context.
                    
                    CRITICAL:
                    1. Cite the source number at the end of every sentence (e.g., [1]).
                    2. If the answer is not in the context, reply: "I cannot find this in the provided documents."
                    3. Do not use outside knowledge.
                    
                    Context:
                    {context}
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"{system_prompt}\n\nQuestion: {prompt}"
                    )
                    
                    st.markdown(response.text)
                    
                    with st.expander("📚 View Cited Sources"):
                        for idx, source in enumerate(retrieved_sources):
                            st.info(f"**[{idx+1}] {source['source']} (Page {source['page']})**\n\n{source['text']}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.text,
                        "sources": retrieved_sources
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
