import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os
from dotenv import load_dotenv

load_dotenv()

def test_agent():
    df = pd.DataFrame({
        "math score": [80, 90, 70, 60],
        "reading score": [85, 95, 75, 65]
    })
    
    import traceback
    # Try with gemini-2.5-flash as currently in app.py
    print("Testing with gemini-2.5-flash...")
    try:
        llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
        )
        response = agent.invoke("Please provide a statistical summary of the main columns.")
        print("Response:", response)
    except Exception as e:
        print(f"Error with 2.5: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_agent()
