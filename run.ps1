# Script to boot both the FastAPI backend and Streamlit frontend

# Ensure the virtual environment is activated if it exists
if (Test-Path -Path .\venv\Scripts\Activate.ps1) {
    . .\venv\Scripts\Activate.ps1
}

Write-Host "Starting FastAPI Backend Server..." -ForegroundColor Green
Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "api:app", "--reload", "--port", "8000"

Write-Host "Waiting 3 seconds for backend to boot..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "Starting Streamlit Frontend Server..." -ForegroundColor Green
streamlit run app.py
