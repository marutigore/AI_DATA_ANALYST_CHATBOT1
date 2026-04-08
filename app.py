"""
Main Application UI.
Streamlit-based frontend for the AI Data Analyst Chatbot.
"""
import os
import streamlit as st
import pandas as pd
from typing import Optional
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import config
from utils.document_loader import load_document
from utils.chunker import chunk_dataframe_to_text
from utils.validator import validate_query

logger = logging.getLogger(__name__)

def initialize_ui() -> None:
    """
    Sets up the initial Streamlit page configuration and header.
    """
    st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")
    
    # Inject Custom CSS for Premium Look
    st.markdown("""
        <style>
        .main-header {
            font-size: 3.5rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #00E5FF, #4B8BFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #1E90FF;
        }
        .stChatMessage p {
            font-size: 1.25rem !important;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if os.path.exists("logo.png"):
        # Display the logo cleanly by itself since the image natively contains the text
        st.image("logo.png", width=280)
    else:
        st.markdown('<p class="main-header">🤖 AI Data Analyst Chatbot</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

def handle_file_upload() -> Optional[pd.DataFrame]:
    """
    Handles file upload widget and parses the uploaded document.
    """
    try:
        # Centralized file uploader with a clean container
        uploaded_file = st.file_uploader("📂 Upload a CSV or Excel dataset to unlock the Chat Interface", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            with st.spinner("Analyzing dataset structure..."):
                df = load_document(uploaded_file, uploaded_file.name)
                
                if df is not None:
                    # Reset memory state if a NEW file is uploaded
                    if "current_file_name" not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
                        st.session_state.current_file_name = uploaded_file.name
                        st.session_state.chunks = chunk_dataframe_to_text(df)
                        # st.session_state.messages = [] # Commented out: Keep history until program close
                        st.session_state.retriever_initialized = False # Force FAISS rebuild
                        st.toast(f'Dataset "{uploaded_file.name}" loaded instantly!', icon='✅')
                        st.balloons()
                    
                    # Collapse profiling details gracefully
                    with st.expander("📊 Data Profiling Metrics & Preview", expanded=False):
                        st.info("The application has parsed your file and securely stored it in ephemeral memory.")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Rows Processed", f"{df.shape[0]:,}")
                        col2.metric("Features (Columns)", df.shape[1])
                        col3.metric("Missing Data Points", df.isna().sum().sum())
                        
                        st.markdown("**Data Snapshot (First 10 records):**")
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    return df
       
            
        return None
    except Exception as e:
        st.error(f"Could not load the file: {str(e)}")
        logger.error(f"File upload exception: {e}")
        return None

def generate_response(df: pd.DataFrame, query: str) -> str:
    """
    Uses LangChain Pandas Agent to process the user query on the dataset.
    """
    try:
        # Check if API Keys are properly set
        if not config.check_keys():
            st.error("⚠️ Missing or invalid API Key. Please verify your .env file.")
            return "Server Error: API Key missing."

        with st.chat_message("assistant"):
            response_container = st.empty()
            response_container.info("🧠 Thinking and writing python code... please wait.")
            
            llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash") 
            
            agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                verbose=True, 
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
            
            response = agent.invoke({"input": query})
            output_text = response.get("output", "No output generated.")
            
            # Display output organically in Chat format
            response_container.markdown(output_text)
            
            # Simple demonstration of citations
            try:
                from utils.embedder import create_embeddings, get_embeddings_model
                from utils.retriever import initialize_vector_store, retrieve_similar
                
                if "chunks" in st.session_state and st.session_state["chunks"]:
                    chunks = st.session_state["chunks"]
                    if not st.session_state.get("retriever_initialized"):
                         embs = create_embeddings(chunks)
                         initialize_vector_store(embs, chunks)
                         st.session_state.retriever_initialized = True
                         
                    model = get_embeddings_model()
                    q_vec = model.encode([query], convert_to_numpy=True).tolist()[0]
                    similar = retrieve_similar(q_vec, 1)
                    if similar:
                        with st.expander("🔍 View Retrieved Context / Citations"):
                            st.caption(f"Semantic Match: {similar[0]['content'][:200]}...")
            except Exception as inner_e:
                pass # Hide citations if FAISS fails locally so UI doesn't visually break
                
            return output_text
            
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        error_msg = f"❌ An internal error occurred while analyzing the data.\n\n`{str(e)}`"
        st.error(error_msg)
        return error_msg

def main() -> None:
    initialize_ui()
    
    # Initialize message history array
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    df = handle_file_upload()
    
    if df is not None:
        st.markdown("---")
        
        # Display existing chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Create persistent bottom chat bar
        prompt = st.chat_input("Ask a question about your dataset! (e.g., 'What is the sum of Revenue grouped by Region?')")
        if prompt:
            try:
                valid_query = validate_query(prompt)
                
                st.session_state.messages.append({"role": "user", "content": valid_query})
                with st.chat_message("user"):
                    st.markdown(valid_query)
                    
                assistant_response = generate_response(df, valid_query)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as ve:
                st.error(str(ve))

if __name__ == "__main__":
    main()
