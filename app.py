import logging
import pandas as pd
import requests
import io
import plotly.express as px
import plotly.io as pio
import streamlit as st

import config
from utils import ui_components

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Luminary AI Workspace", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

API_URL = "http://127.0.0.1:8000/api"

def inject_styles():
    st.markdown(ui_components.get_design_tokens(), unsafe_allow_html=True)
    # Streamlit button active state hacking
    active_btn_css = """
    <style>
    .active-btn button {
        background-color: rgba(255,255,255,0.05) !important;
        color: var(--text-main) !important;
        border-left-color: var(--primary) !important;
    }
    </style>
    """
    st.markdown(active_btn_css, unsafe_allow_html=True)

def render_sidebar():
    st.sidebar.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 2rem;">
            <div style="width:32px; height:32px; background:var(--primary); border-radius:8px; display:flex; align-items:center; justify-content:center;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            </div>
            <div>
                <h3 style="margin:0; font-size:1.1rem; color:white;">Luminary AI</h3>
                <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:0.05em;">WORKSPACE</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("New Analysis", type=["csv", "xlsx"], label_visibility="collapsed")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # State routing via buttons
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "CURRENT CHAT"

    # Navigation Setup
    if st.sidebar.button("💬 CURRENT CHAT", use_container_width=True):
        st.session_state.current_view = "CURRENT CHAT"
        
    if st.sidebar.button("💿 DATASET EXPLORER", use_container_width=True):
        st.session_state.current_view = "DATASET EXPLORER"
        
        
    # Removed unrequired disabled buttons to keep the sidebar clean

    return uploaded_file

@st.cache_data
def upload_to_backend(bytes_data, file_name):
    try:
        files = {"file": (file_name, bytes_data)}
        response = requests.post(f"{API_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to connect to backend API: {e}")
        return None

def render_chat_view():
    # Conditionally force this view to use a ChatGPT-style centered, auto-adjusting layout
    st.markdown("""
        <style>
        [data-testid="block-container"] {
            max-width: 850px !important;
            margin: 0 auto !important;
        }
        </style>
    """, unsafe_allow_html=True)

    file_name = st.session_state.get('filename', 'None (Awaiting Upload)')
    session_id = st.session_state.get('session_id')

    if session_id:
        st.markdown(f"<div style='text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-bottom: 30px; margin-top: 10px;'>Active Dataset: <strong>{file_name}</strong></div>", unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        if not session_id:
            ui_components.render_hero_empty_state()
        else:
            if len(st.session_state.messages) == 0:
                st.markdown("<p style='color:var(--text-muted); font-style:italic; text-align: center; margin-bottom: 2rem;'>Ask Luminary a question about your data...</p>", unsafe_allow_html=True)
                st.markdown("<div style='text-align:center; margin-bottom: 10px; font-size: 0.85rem; color: var(--text-muted);'>Suggested Queries:</div>", unsafe_allow_html=True)
                suggestions = st.session_state.get('suggested_queries', [])
                if not suggestions:
                    # Fallback if no suggestions are available
                    suggestions = [
                        {"icon": "📊", "title": "Show Data Summary", "query": "Please provide a statistical summary of the main columns."},
                        {"icon": "📈", "title": "Identify Key Trends", "query": "What are the most significant trends or outliers in this dataset?"},
                        {"icon": "🧠", "title": "Data Quality Check", "query": "Analyze the data quality, including missing values and inconsistencies."}
                    ]
                
                cols = st.columns(min(len(suggestions), 3))
                for i, sugg in enumerate(suggestions[:3]):
                    with cols[i]:
                        btn_text = f"{sugg.get('icon', '💡')} {sugg.get('title', 'Query')}"
                        if st.button(btn_text, use_container_width=True):
                            prompt = sugg.get('query')
                            st.session_state.messages.append({"role": "user", "content": prompt})
                            st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # Render historical plots if they exist
                    if "plot_json" in msg and msg["plot_json"] is not None:
                        try:
                            fig = pio.from_json(msg["plot_json"])
                            st.plotly_chart(ui_components.style_luminary_figure(fig), use_container_width=True, theme=None)
                        except:
                            pass

    if session_id:
        if prompt := st.chat_input("Ask Luminary..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    if session_id and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["content"]
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing parameters via Backend API..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/chat",
                            json={"session_id": session_id, "prompt": prompt}
                        )
                        if response.status_code == 429:
                            st.warning("⚠️ **API Quota Exceeded.** Your free-tier Gemini limit has been reached. Please wait a few minutes and try again, or visit [Google AI Studio](https://ai.dev/rate-limit) to check your usage.")
                            st.session_state.messages.pop()
                        elif not response.ok:
                            try:
                                error_msg = response.json().get("detail", response.text)
                            except:
                                error_msg = response.text
                            raise Exception(error_msg)
                        else:
                            data = response.json()
                            st.markdown(data["response"])
                            
                            plot_json_str = data.get("plot_json")
                            if plot_json_str:
                                try:
                                    plot_fig = pio.from_json(plot_json_str)
                                    styled = ui_components.style_luminary_figure(plot_fig)
                                    st.plotly_chart(styled, use_container_width=True, theme=None)
                                except Exception as json_e:
                                    st.error(f"Failed to render chart from API: {json_e}")

                            # Add message to history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": data["response"],
                                "plot_json": plot_json_str
                            })
                    except requests.exceptions.Timeout:
                        st.warning("⏱️ **Connection timed out.** The backend server did not respond in time. Please check the server is running and try again.")
                        st.session_state.messages.pop()
                    except Exception as e:
                        st.error(f"Analysis Error: {str(e)}")
                        st.session_state.messages.pop()

def render_dataset_explorer_view():
    if not st.session_state.get('session_id'):
        ui_components.render_dataset_empty_state()
        return

    metrics = st.session_state.get('metrics', {})
    rows = metrics.get('rows', 0)
    cols = metrics.get('cols', 0)
    memory = metrics.get('memory_mb', 0)
    missing = metrics.get('missing', 0)

    # Header Row: KPIs
    k_col1, k_col2, k_col3 = st.columns(3, gap="medium")
    with k_col1:
        ui_components.render_kpi("⚏ TOTAL RECORDS", f"{rows:,}", f"{memory:.1f} MB Total Size", trend="Live Sync" if rows > 0 else "")
    with k_col2:
        ui_components.render_kpi("⚡ FEATURE COLUMNS", f"{cols}", "Detected variables", trend="Active")
    with k_col3:
        ui_components.render_kpi("★ MISSING VALUES", f"{missing}", "Dataset Integrity check", trend="Warning" if missing > 0 else "Pristine")

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    file_name = st.session_state.get("filename", "")
    st.markdown(f"<div style='font-size:1.1rem; font-weight:600; color:white; margin-bottom:10px;'>{file_name} Preview</div>", unsafe_allow_html=True)
    
    preview_json = st.session_state.get('preview_json')
    if preview_json:
        # Use StringIO to avoid specific warnings with direct read_json strings
        df_preview = pd.read_json(io.StringIO(preview_json))
        st.dataframe(df_preview, use_container_width=True)

def main():
    inject_styles()
    uploaded_file = render_sidebar()
    ui_components.render_top_nav()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Handle Upload - only run when new file or not yet uploaded
    if uploaded_file is not None and ('uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name):
        bytes_data = uploaded_file.getvalue()
        with st.spinner("Connecting to FastAPI Backend..."):
            backend_data = upload_to_backend(bytes_data, uploaded_file.name)
            if backend_data:
                st.session_state.session_id = backend_data['session_id']
                st.session_state.filename = backend_data['filename']
                st.session_state.metrics = backend_data['metrics']
                st.session_state.preview_json = backend_data['preview_json']
                st.session_state.suggested_queries = backend_data.get('suggestions', [])
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.messages = [] # reset chat for new file

    # ROUTING
    if st.session_state.current_view == "CURRENT CHAT":
        render_chat_view()
    elif st.session_state.current_view == "DATASET EXPLORER":
        render_dataset_explorer_view()

if __name__ == "__main__":
    main()
