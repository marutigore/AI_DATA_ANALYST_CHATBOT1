import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def get_design_tokens():
    return """
    <style>
        /* Luminary specific adjustments */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background: transparent;}
        
        :root {
            /* Luminary Palettes */
            --bg-deep: #0F111A;
            --bg-card: #1A1D27;
            --bg-card-hover: #222634;
            --primary: #4F46E5;        /* Indigo / Blue button */
            --primary-glow: rgba(79, 70, 229, 0.4);
            --accent-green: #2DD4BF;   /* Neon Sea Green */
            --accent-purple: #8B5CF6;  /* Neon Purple */
            --text-main: #F8FAFC;
            --text-muted: #94A3B8;
            --border: #2E3344;
            --border-purple: rgba(139, 92, 246, 0.3);
            --radius-md: 10px;
            --radius-lg: 16px;
            --shadow-glow: 0 0 20px rgba(139, 92, 246, 0.15);
        }

        .stApp {
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-deep);
            color: var(--text-main);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #12141C !important;
            border-right: 1px solid var(--border);
            padding-top: 1rem;
        }

        /* The Main Header */
        .luminary-top-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0 20px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
        }
        
        .nav-links {
            display: flex;
            gap: 24px;
            font-size: 0.9rem;
            color: var(--text-muted);
            font-weight: 500;
        }
        
        .nav-link.active {
            color: var(--primary);
            border-bottom: 2px solid var(--primary);
            padding-bottom: 4px;
        }
        
        .search-bar {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 16px;
            width: 300px;
            color: var(--text-muted);
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Big Blue Button overriding st.file_uploader or st.button */
        div[data-testid="stFileUploader"] section, button[kind="primary"] {
            background: linear-gradient(135deg, #4F46E5, #3B82F6) !important;
            color: white !important;
            border-radius: var(--radius-md) !important;
            border: none !important;
            box-shadow: 0 4px 15px var(--primary-glow) !important;
            font-weight: 600 !important;
            transition: all 0.2s !important;
        }
        
        button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px var(--primary-glow) !important;
        }

        /* Container overrides for absolute left alignment */
        section[data-testid="stSidebar"] .block-container, 
        [data-testid="stSidebarUserContent"] {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }

        /* Sidebar Navigation Buttons - AGGRESSIVE LEFT ALIGNMENT */
        div[data-testid="stSidebar"] button {
            background-color: transparent !important;
            background: transparent !important;
            color: var(--text-muted) !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            border-radius: 6px !important;
            width: 100% !important;
            padding: 8px 5px !important;
            margin: 0 !important;
            display: flex !important;
            justify-content: flex-start !important;
            align-items: center !important;
        }

        div[data-testid="stSidebar"] div.stButton {
            margin-left: 0 !important;
            padding-left: 0 !important;
            width: 100% !important;
        }

        div[data-testid="stSidebar"] button * {
            display: flex !important;
            justify-content: flex-start !important;
            text-align: left !important;
            font-weight: 600 !important;
            letter-spacing: 0.05em !important;
            margin-left: 0 !important;
            padding-left: 0 !important;
        }
        
        [data-testid="stSidebar"] button[kind="secondary"]:hover,
        [data-testid="stSidebar"] button[kind="secondary"]:active,
        [data-testid="stSidebar"] button[kind="secondary"]:focus {
            background-color: rgba(255,255,255,0.05) !important;
            color: var(--text-main) !important;
            border-color: transparent !important;
            border-left-color: var(--primary) !important;
        }

        /* Auto-Insight Card */
        .insight-card {
            background: linear-gradient(180deg, rgba(26, 29, 39, 0.9) 0%, rgba(30, 27, 43, 0.9) 100%);
            border: 1px solid var(--border-purple);
            border-radius: var(--radius-md);
            padding: 20px;
            box-shadow: var(--shadow-glow);
            margin-bottom: 20px;
        }
        
        .insight-header {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: var(--accent-purple);
            text-transform: uppercase;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .insight-body {
            font-size: 0.95rem;
            line-height: 1.5;
            color: var(--text-main);
            font-style: italic;
        }

        /* KPI Card */
        .luminary-kpi {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 20px;
            height: 100%;
        }
        .kpi-title {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
        }
        .kpi-value {
            font-size: 2.2rem;
            font-weight: 600;
            color: var(--text-main);
        }
        .kpi-sub {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 8px;
        }
        .trend-up {
            color: var(--accent-green);
            background: rgba(45, 212, 191, 0.1);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        /* Streamlit overrides */
        .stMarkdown p {
            color: var(--text-muted);
        }
        h1, h2, h3 {
            color: var(--text-main) !important;
            font-weight: 600 !important;
        }

        /* Chat Message Styling */
        div[data-testid="stChatMessage"] {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1rem;
            margin-bottom: 12px;
        }
        div[data-testid="stChatMessage"] * {
            color: var(--text-main) !important;
        }
        
        /* Hero Empty State */
        .hero-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 4rem 2rem;
            text-align: center;
            animation: fadeIn 1s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #F8FAFC, #94A3B8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            color: var(--text-muted);
            max-width: 500px;
            margin-bottom: 3rem;
            line-height: 1.6;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            width: 100%;
            max-width: 800px;
        }
        .feature-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            text-align: left;
        }
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, var(--primary), var(--accent-purple));
            opacity: 0;
            transition: opacity 0.3s;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-glow);
            border-color: var(--border-purple);
        }
        .feature-card:hover::before {
            opacity: 1;
        }
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        .feature-title {
            color: var(--text-main);
            font-weight: 600;
            margin-bottom: 0.5rem;
            font-size: 1.05rem;
        }
        .feature-desc {
            color: var(--text-muted);
            font-size: 0.85rem;
            line-height: 1.5;
        }
    </style>
    """

def render_top_nav():
    html = """
    <div class="luminary-top-nav">
        <div style="flex: 1;">
            <h2 style="margin: 0; color: white;">Luminary <span style="font-weight:400; color:#94A3B8;">AI</span></h2>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_kpi(title, value, sub, trend=None):
    trend_html = f'<span class="trend-up">{trend}</span>' if trend else ""
    html = f"""
    <div class="luminary-kpi">
        <div class="kpi-title">
            <span>{title}</span>
            {trend_html}
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_insight(label: str, title: str, text: str, highlight_color: str = "#8B5CF6", border_color: str = "#4C1D95", time_str: str = "Just now"):
    html = f"""
    <div class="insight-card" style="border-color: {border_color}; box-shadow: 0 0 20px rgba(139, 92, 246, 0.05);">
        <div class="insight-header" style="color: {highlight_color};">
            <span>✦ {label}</span>
            <span style="color: #64748B; font-weight: 400;"> • {time_str}</span>
        </div>
        <div class="insight-body">
            <strong>{title}</strong><br>
            {text}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def style_luminary_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the Luminary dark theme to a Plotly figure in a single fast update_layout call."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,32,0.6)",
        font=dict(color="#94A3B8", size=12),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False, tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False, tickfont=dict(size=10)
        ),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10))
    )
    return fig

def render_hero_empty_state():
    html = """
    <div class="hero-container">
        <div class="hero-subtitle">Upload your CSV or Excel dataset in the sidebar to unlock AI-powered insights, automated visualizations, and conversational data exploration.</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_dataset_empty_state():
    html = """
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height: 50vh; text-align:center; animation: fadeIn 0.5s ease-out;">
        <div style="font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.5;">🗃️</div>
        <h3 style="color: var(--text-main); margin-bottom: 0.5rem;">No Dataset Connected</h3>
        <p style="color: var(--text-muted); max-width: 400px; line-height: 1.5;">Please upload a file from the sidebar to view its schema, profile, and raw data preview.</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
