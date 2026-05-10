import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from src.pipeline.hybrid_pipeline import HybridPipelineFinal

st.set_page_config(page_title="Lumora Analytics Assistant", page_icon="🚀", layout="wide")

@st.cache_resource
def load_pipeline():
    return HybridPipelineFinal()

pipeline = load_pipeline()

st.title("🚀 Lumora Analytics Enterprise Assistant")
st.markdown("**Hybrid RAG System** — Structured + Semantic Search + ML Router")

with st.sidebar:
    st.success("✅ System Online")
    st.write("**Router:** Fine-tuned DistilBERT")
    st.write("**LLM:** Llama 3.2")
    st.caption("Data Scientist Portfolio Project")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about the company..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = pipeline.handle_query(prompt)
            answer = result.get("answer", "Sorry, I couldn't generate a response.")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Quick Examples (without Aisha Patel)
st.divider()
st.subheader("Quick Examples")
cols = st.columns(3)

with cols[0]:
    if st.button("DevOps Team Members"):
        st.session_state.messages.append({"role": "user", "content": "List all members of the DevOps team"})
        st.rerun()

with cols[1]:
    if st.button("Data Retention Policy"):
        st.session_state.messages.append({"role": "user", "content": "What is our data retention policy?"})
        st.rerun()

with cols[2]:
    if st.button("StreamAPI Security"):
        st.session_state.messages.append({"role": "user", "content": "What security policies apply to StreamAPI?"})
        st.rerun()