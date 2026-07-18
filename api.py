import os
import uuid
import json
import logging
from typing import Optional, Dict

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.document_loader import load_document

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="Luminary AI Backend")

# In-memory session store: mapping session_id to an AgentExecutor
SESSION_STORE: Dict[str, dict] = {}

class ChatRequest(BaseModel):
    session_id: str
    prompt: str

class ChatResponse(BaseModel):
    response: str
    plot_json: Optional[str] = None

def init_agent(df: pd.DataFrame, session_id: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    chart_filename = f"temp_chart_{session_id[:8]}.json"
    custom_prefix = f"""You are Luminary AI, a fast, expert data analyst.

For ALL chart/plot/graph requests:
1. Use ONLY `plotly.express` (it is pre-imported as `px`).
2. Create the figure with a single, concise `px.*` call.
3. NEVER call `fig.show()`.
4. Save the figure to a file: `fig.write_json("{chart_filename}")`
5. Print ONLY the word: CHART_DONE
6. Your final answer must be ONLY: "I have generated the chart for you."

For ALL non-chart requests:
- Answer concisely using Python pandas operations on `df`.
- Do NOT import anything — pandas (pd), numpy (np), and plotly.express (px) are already available.
- Keep answers short and factual.
"""
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        prefix=custom_prefix,
        max_iterations=5,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

from fastapi.concurrency import run_in_threadpool

def generate_suggestions(df: pd.DataFrame) -> list:
    """
    Generate smart, context-aware suggestions instantly using pandas — no API calls, no quota cost.
    Inspects column dtypes and picks the most relevant analytical queries for the dataset.
    """
    suggestions = []
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

    # Suggestion 1: Always offer a statistical summary
    if numeric_cols:
        col_list = ', '.join(numeric_cols[:3])
        suggestions.append({
            "icon": "📊",
            "title": "Data Summary",
            "query": f"Give me a statistical summary (mean, median, std, min, max) for the columns: {col_list}."
        })
    else:
        suggestions.append({
            "icon": "📊",
            "title": "Data Overview",
            "query": "Please provide a full overview of this dataset including its shape, column names, and data types."
        })

    # Suggestion 2: Time-series trend if date col exists, else distribution of top numeric col
    if date_cols and numeric_cols:
        suggestions.append({
            "icon": "📅",
            "title": "Trend Over Time",
            "query": f"Plot a line chart showing how '{numeric_cols[0]}' changes over '{date_cols[0]}' over time."
        })
    elif numeric_cols:
        suggestions.append({
            "icon": "📈",
            "title": "Top Outliers",
            "query": f"What are the top 5 largest and 5 smallest values in '{numeric_cols[0]}'? Are there any outliers?"
        })
    else:
        suggestions.append({
            "icon": "📈",
            "title": "Identify Key Trends",
            "query": "What are the most significant trends or patterns in this dataset?"
        })

    # Suggestion 3: Categorical breakdown or data quality check
    if cat_cols:
        suggestions.append({
            "icon": "🏷️",
            "title": "Category Breakdown",
            "query": f"Show me a bar chart of the top 10 most frequent values in the '{cat_cols[0]}' column."
        })
    else:
        suggestions.append({
            "icon": "🧠",
            "title": "Data Quality Check",
            "query": "Analyze the data quality: how many missing values are there per column, and are there any duplicate rows?"
        })

    return suggestions[:3]

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        import io
        df = load_document(io.BytesIO(contents), file.filename)
        
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        agent = init_agent(df, session_id)
        SESSION_STORE[session_id] = {
            "agent": agent,
            "filename": file.filename,
            "chart_filename": f"temp_chart_{session_id[:8]}.json",
        }
        
        # Compute KPIs for Streamlit
        metrics = {
            "rows": len(df),
            "cols": len(df.columns),
            "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
            "missing": int(df.isna().sum().sum())
        }
        
        # Generate dynamic suggestions based on the dataset
        suggestions = await run_in_threadpool(generate_suggestions, df)

        return {
            "session_id": session_id,
            "filename": file.filename,
            "metrics": metrics,
            "preview_json": df.head(40).to_json(),
            "suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_data(request: ChatRequest):
    session = SESSION_STORE.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please re-upload the dataset.")
    
    agent = session["agent"]
    
    try:
        # Run the synchronous agent in a thread pool so it doesn't block the event loop.
        # No timeout — let the AI take as long as it needs.
        response = await run_in_threadpool(agent.invoke, request.prompt)
        output_text = response.get("output", "")

        # Fast path: read chart from disk if agent wrote it
        plot_json_str = None
        chart_filename = session.get("chart_filename", f"temp_chart_{request.session_id[:8]}.json")
        if os.path.exists(chart_filename):
            try:
                with open(chart_filename, "r", encoding="utf-8") as f:
                    plot_json_str = f.read()
                os.remove(chart_filename)
                # Clean up the output text for chart responses
                output_text = "I have generated the chart for you."
            except Exception as e:
                logger.error(f"Failed to read {chart_filename}: {e}")
                
        return ChatResponse(
            response=output_text,
            plot_json=plot_json_str
        )
    except Exception as e:
        error_str = str(e)
        # Give a friendly message on quota errors
        if "429" in error_str or "quota" in error_str.lower() or "ResourceExhausted" in error_str:
            raise HTTPException(status_code=429, detail="API quota exceeded. Please wait a moment and try again, or check your Gemini API plan at https://ai.dev/rate-limit")
        logger.error(f"Chat error: {error_str}")
        raise HTTPException(status_code=500, detail=error_str)

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
