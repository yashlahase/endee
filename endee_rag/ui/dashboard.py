import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Endee RAG - AI Semantic Search",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00ffa3;
        color: #0e1117;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00d187;
        color: #0e1117;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .css-1n76uvr {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
    }
    h1 {
        color: #00ffa3;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Settings")
    index_name = st.text_input("Index Name", value="my_docs")
    st.divider()
    st.markdown("### About Endee RAG")
    st.info("This system uses the Endee Vector Database for high-performance retrieval and Google Gemini AI for answering.")
    
    # Health check in sidebar
    try:
        health = requests.get(f"{API_URL}/health").json()
        if health.get("endee_connected"):
            st.success("Endee Vector DB: Connected")
        else:
            st.error("Endee Vector DB: Disconnected")
    except:
        st.error("API Server: Offline")

# Header
st.title("🚀 Endee RAG Dashboard")
st.subheader("Semantic Search & Retrieval Augmented Generation")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF or Text file", type=["pdf", "txt"])
    
    if st.button("Index Document"):
        if uploaded_file and index_name:
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = {"index_name": index_name}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files, data=data)
                    if response.status_code == 200:
                        st.success(f"Indexed {uploaded_file.name} successfully!")
                        st.balloons()
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
        else:
            st.warning("Please upload a file and specify an index name.")

with col2:
    st.markdown("### 🔍 Ask Questions")
    query = st.text_input("Enter your question based on uploaded documents:")
    
    if st.button("Ask AI"):
        if query and index_name:
            with st.spinner("Retrieving context and generating answer..."):
                payload = {"query": query, "index_name": index_name, "k": 3}
                try:
                    response = requests.post(f"{API_URL}/query", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown("#### AI Answer")
                        st.write(data["answer"])
                        
                        with st.expander("View Source Context"):
                            for i, source in enumerate(data["sources"]):
                                st.markdown(f"**Source {i+1}** (ID: {source.get('id')})")
                                st.code(source.get("meta", "No content"), language="text")
                    else:
                        st.error("Could not find relevant documents or index does not exist.")
                except Exception as e:
                    st.error(f"Error during query: {str(e)}")
        else:
            st.warning("Please enter a question.")

# Footer
st.divider()
st.markdown("<p style='text-align: center; color: #8b949e;'>Built with Endee Vector Database, FastAPI, and Streamlit</p>", unsafe_allow_html=True)
