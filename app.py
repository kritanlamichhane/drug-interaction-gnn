"""
Root entry point for the Streamlit application.
Runs app/streamlit_app.py directly.
"""
import os
import sys
import runpy

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
APP_DIR = os.path.join(ROOT_DIR, "app")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Run streamlit_app.py directly
target_script = os.path.join(APP_DIR, "streamlit_app.py")
runpy.run_path(target_script, run_name="__main__")
