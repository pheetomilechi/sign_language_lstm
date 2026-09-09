import streamlit_app
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "sign_language_lstm")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


if __name__ == "__main__":
    streamlit_app.main()
