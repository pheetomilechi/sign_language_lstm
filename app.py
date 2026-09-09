import importlib.util
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "sign_language_lstm")
APP_FILE = os.path.join(APP_DIR, "streamlit_app.py")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

spec = importlib.util.spec_from_file_location("streamlit_app", APP_FILE)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load Streamlit app module from {APP_FILE}")

streamlit_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streamlit_app)


if __name__ == "__main__":
    streamlit_app.main()
