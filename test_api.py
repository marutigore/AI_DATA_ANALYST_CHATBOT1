import requests
import json
import io

API_URL = "http://127.0.0.1:8000/api"

# Create dummy csv
csv_content = """id,math score,reading score
1,80,85
2,90,95
3,70,75
4,60,65
"""

def test():
    # Upload
    files = {"file": ("dummy.csv", csv_content)}
    print("Uploading...")
    response = requests.post(f"{API_URL}/upload", files=files)
    if not response.ok:
        print("Upload failed:", response.status_code, response.text)
        return
    data = response.json()
    session_id = data["session_id"]
    print("Session ID:", session_id)
    
    # Chat
    prompt = "Please provide a statistical summary of the main columns."
    print("Chatting...")
    response = requests.post(f"{API_URL}/chat", json={"session_id": session_id, "prompt": prompt})
    if not response.ok:
        print("Chat failed:", response.status_code, response.text)
        return
    
    print("Chat response:", response.json())

if __name__ == "__main__":
    test()
