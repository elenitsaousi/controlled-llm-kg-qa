@echo off
cd /d "%~dp0"
echo Starting True Demand KG QA Streamlit UI...
echo This UI does not require Node.js or npm.
echo Open http://localhost:8501 if the browser does not open automatically.
python -m streamlit run app.py --server.port 8501
