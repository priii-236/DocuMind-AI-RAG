"""
Convenience entry point.

    python run.py

launches the Streamlit app the same way `streamlit run app/streamlit_app.py` does.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)]))
