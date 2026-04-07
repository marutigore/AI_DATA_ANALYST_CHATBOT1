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
    
    Args:
        None
        
    Returns:
        None
        
    Example:
        >>> initialize_ui()
    """
    st.set_page_config(page_title="AI Data Analyst Chatbot", page_icon="📊", layout="wide")
    st.title("📊 AI Data Analyst Chatbot")
    st.write("Upload a CSV/Excel file and ask analytical questions to generate insights and charts!")

def handle_file_upload() -> Optional[pd.DataFrame]:
    """
    Handles file upload widget and parses the uploaded document.
    
    Args:
        None
    
    Returns:
        Optional[pd.DataFrame]: The loaded dataframe, or None if failed/empty.
        
    Example:
        >>> df = handle_file_upload()
    """
    try:
        uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            with st.spinner("Loading and profiling data..."):
                df = load_document(uploaded_file, uploaded_file.name)
                
                if df is not None:
                    st.sidebar.success("File uploaded successfully!")
                    st.subheader("Data Profiling")
                    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                    st.write("**Data Preview:**")
                    st.dataframe(df.head())
                    
                    # RAG initialization (fallback / documentation validation test coverage)
                    chunks = chunk_dataframe_to_text(df)
                    st.session_state["chunks"] = chunks
                    
                    return df
        return None
    except Exception as e:
        st.error(f"Could not load the file: {str(e)}")
        logger.error(f"File upload exception: {e}")
        return None

def generate_response(df: pd.DataFrame, query: str) -> None:
    """
    Uses LangChain Pandas Agent to process the user query on the dataset.
    
    Args:
        df (pd.DataFrame): The dataset context.
        query (str): Validated user question.
        
    Returns:
        None
        
    Example:
        >>> generate_response(df, "What is the highest value in column X?")
    """
    try:
        # Check if API Keys are properly set before dispatching call
        if not config.check_keys():
            st.error("Missing or invalid GOOGLE_API_KEY. Please verify your .env file.")
            return

        with st.spinner("Analyzing data and generating code..."):
            # We initialize a Gemini model to power the code-generation agent logic
            llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash") # Using Flash for higher free tier limits
            
            # Create pandas dataframe agent with allow_dangerous_code due to exec() security changes
            # We allow it since it runs purely in memory on the user's uploaded dataframe
            agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                verbose=True, 
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
            
            response = agent.invoke({"input": query})
            
            st.write("### Analysis Results")
            st.write(response.get("output", "No output generated."))
            
            # Simple demonstration of citations relying on our vector store
            try:
                from utils.embedder import create_embeddings, get_embeddings_model
                from utils.retriever import initialize_vector_store, retrieve_similar
                
                if "chunks" in st.session_state and st.session_state["chunks"]:
                    chunks = st.session_state["chunks"]
                    # If store not initialized, init it here
                    if "retriever_initialized" not in st.session_state:
                         embs = create_embeddings(chunks)
                         initialize_vector_store(embs, chunks)
                         st.session_state["retriever_initialized"] = True
                         
                    # Create query embedding via loaded model
                    model = get_embeddings_model()
                    q_vec = model.encode([query], convert_to_numpy=True).tolist()[0]
                    # Retrieve the most highly matched document text
                    similar = retrieve_similar(q_vec, 1)
                    if similar:
                        st.caption("Citations:")
                        st.text(f"Relevant Data Snippet: {similar[0]['content'][:100]}...")
            except Exception as inner_e:
                logger.warning(f"Failed to generate citation: {inner_e}")
                
            
            st.download_button(
                label="Download Result",
                data=str(response.get("output", "")),
                file_name="analysis_result.txt"
            )
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        st.error(f"An error occurred while analyzing the data: {e}")

def main() -> None:
    """
    Main application loop.
    
    Args:
        None
        
    Returns:
        None
    """
    initialize_ui()
    
    df = handle_file_upload()
    
    if df is not None:
        query = st.chat_input("Ask a question about your data (e.g., 'Show me a bar chart of sales by region').")
        if query:
            try:
                # Validating input strictly to prevent malformed operations
                valid_query = validate_query(query)
                generate_response(df, valid_query)
            except ValueError as ve:
                st.error(str(ve))

if __name__ == "__main__":
    main()
