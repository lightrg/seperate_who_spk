#!/bin/bash
# Web App Run Script (Streamlit)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/.."

# Launch the main interface of the system.
echo "Starting Web Application via Streamlit..."
streamlit run app_streamlit.py "$@"
