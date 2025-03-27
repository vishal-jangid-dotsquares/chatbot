import subprocess
import time

# Start FastAPI backend
fastapi_process = subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

# Give FastAPI some time to start
time.sleep(3)

# Start Streamlit frontend
streamlit_process = subprocess.Popen(["streamlit", "run", "streamlit.py", "--server.headless", "true"])

# Wait for both processes to complete
try:
    fastapi_process.wait()
    streamlit_process.wait()
except KeyboardInterrupt:
    fastapi_process.terminate()
    streamlit_process.terminate()
