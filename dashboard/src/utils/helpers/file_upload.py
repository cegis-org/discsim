import streamlit as st
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from src.utils.helpers.fetch_files import fetch_files_from_api,fetch_file_from_api

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

UPLOAD_FILE_ENDPOINT = f"{API_BASE_URL}/upload_file"

def handle_file_upload(file_option, category):
    uploaded_file = None

    if file_option == "Upload a file":

        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV (UTF-8 encoded) to begin analysis",
            type="csv",
            help="**Please ensure the CSV is ready for analysis: such as data starting from the first row. If you have data in any other format, please convert to CSV (UTF-8 encoded) to begin analysis",
        )

        if uploaded_file is not None:
            uploaded_file.seek(0)
            files = {"file": uploaded_file}
            data = {"category": category}
            # Make API call to upload the file
            response = requests.post(UPLOAD_FILE_ENDPOINT, files=files, data=data)

            if response.status_code == 200:
                file_id = response.json()["id"]
                st.session_state.uploaded_file_id = file_id
                st.session_state.current_file_name = uploaded_file.name
                uploaded_file = fetch_file_from_api(file_id)
                st.session_state.uploaded_file = uploaded_file
            elif response.status_code == 409:
                st.warning(f"A file with this name already exists in {category}. Please upload a different file.")
                return None
            else:
                st.error("Failed to upload file.")
                return None

    # Handle previously uploaded file selection
    elif file_option == "Select a previously uploaded file":
        files = fetch_files_from_api(category)
        if not files:
            st.warning(f"No files have been uploaded yet in {category}.")
            return None
        
        default_option = "Select a file"
        # Format file names with upload datetime
        file_options = [default_option] + [
            f"{file['filename']}: {(datetime.fromisoformat(file['upload_datetime']) + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d')}"
            for file in files
        ]
        selected_option = st.sidebar.selectbox(f"Select a previously uploaded file ({len(file_options)} files)", file_options)

        if selected_option == default_option:
            return None
        else: 
            if selected_option:
                selected_filename = selected_option.split(": ")[0]
                # Find the selected file's ID
                try:
                    file_id = next(
                        file["id"]
                        for file in files
                        if file["filename"] == selected_filename
                    )
                    st.session_state.uploaded_file_id = file_id
                    uploaded_file = fetch_file_from_api(file_id)
                    st.session_state.uploaded_file = uploaded_file
                except StopIteration:
                    st.error(f"No file found with the name '{selected_filename}' in {category}. Please try again.")
                    return None

    return uploaded_file
