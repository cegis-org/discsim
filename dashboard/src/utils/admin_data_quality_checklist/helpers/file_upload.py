import streamlit as st
import requests
import sys
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from src.utils.admin_data_quality_checklist.helpers.fetch_files import fetch_file_from_api

load_dotenv()

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE_URL = os.getenv("API_BASE_URL")

UPLOAD_FILE_ENDPOINT = f"{API_BASE_URL}/upload_file"

def handle_file_upload(file_option):
    uploaded_file = None
    
    if file_option == "Upload a new file":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file to begin analysis", type="csv", help="**Please ensure the CSV is ready for analysis: such as data starting from the first row. If you have data in any other format, please convert to CSV to begin analysis")

        if uploaded_file is not None:
            if "current_file_name" not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                response = requests.post(UPLOAD_FILE_ENDPOINT, files=files)

                if response.status_code == 200:
                    file_id = response.json()["id"]
                    st.session_state.uploaded_file_id = file_id
                    st.session_state.current_file_name = uploaded_file.name
                    uploaded_file = fetch_file_from_api(file_id)
                elif response.status_code == 409:
                    st.warning("A file with this name already exists. Please upload a different file.")
                    return None
                else:
                    st.error("Failed to upload file.")
                    return None

    elif file_option == "Select a previously uploaded file":
        response = requests.get(f"{API_BASE_URL}/list_files")
        if response.status_code == 200:
            files = response.json()
            if not files:
                st.warning("No files have been uploaded yet.")
                return None

            # Format file names with upload datetime
            file_options = [f"{file['filename']}: {(datetime.fromisoformat(file['upload_datetime']) + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d')}" for file in files]
            selected_option = st.sidebar.selectbox(f"Select a previously uploaded file ({len(file_options)} files)", file_options)

            if selected_option:
                selected_filename = selected_option.split(": ")[0]
                try:
                    file_id = next(file["id"] for file in files if file["filename"] == selected_filename)
                    st.session_state.uploaded_file_id = file_id
                    uploaded_file = fetch_file_from_api(file_id)
                except StopIteration:
                    st.error(f"No file found with the name '{selected_filename}'. Please try again.")
                    return None
        else:
            st.error("Failed to retrieve file list.")
            return None

    return uploaded_file