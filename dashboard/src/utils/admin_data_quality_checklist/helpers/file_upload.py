import streamlit as st
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from src.utils.admin_data_quality_checklist.helpers.fetch_files import fetch_file_from_api,get_file

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

UPLOAD_FILE_ENDPOINT = f"{API_BASE_URL}/upload_file"

# Cache the file listing function to prevent redundant calls to the server
@st.cache_data
def fetch_files_from_api(category):
    params = {"category": category}
    response = requests.get(f"{API_BASE_URL}/list_files", params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to retrieve file list. Status code: {response.status_code}")
        return []

def handle_file_upload(file_option, category):
    uploaded_file = None

    if file_option == "Upload a new file":

        if 'uploaded_file_id' in st.session_state:
            del st.session_state['uploaded_file_id']
        if 'uploaded_file' in st.session_state:
            del st.session_state['uploaded_file']
        if 'file_list' in st.session_state:
            del st.session_state['file_list']
        if 'current_file_name' in st.session_state:
            del st.session_state['current_file_name']
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file to begin analysis",
            type="csv",
            help="**Please ensure the CSV is ready for analysis: such as data starting from the first row. If you have data in any other format, please convert to CSV to begin analysis",
        )

        if uploaded_file is not None:
            # Check if the file has already been uploaded in the session state
            if (
                "current_file_name" not in st.session_state
                or st.session_state.current_file_name != uploaded_file.name
            ):
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                data = {"category": category}
                # Make API call to upload the file
                response = requests.post(UPLOAD_FILE_ENDPOINT, files=files, data=data)

                if response.status_code == 200:
                    file_id = response.json()["id"]
                    st.session_state.uploaded_file_id = file_id
                    st.session_state.current_file_name = uploaded_file.name
                    uploaded_file = get_file(file_id)
                    st.session_state.uploaded_file = uploaded_file # Store file in session state
                    del st.session_state['file_list'] #delete the cached filelist
                elif response.status_code == 409:
                    st.warning(
                        f"A file with this name already exists in {category}. Please upload a different file."
                    )
                    return None
                else:
                    st.error("Failed to upload file.")
                    return None

    # Handle previously uploaded file selection
    elif file_option == "Select a previously uploaded file":
         # Fetch file list from session state if already fetched
        if 'file_list' in st.session_state:
            files = st.session_state.file_list
        else:
            # Fetch file list from API and cache it
            files = fetch_files_from_api(category)
            st.session_state.file_list = files  # Save to session state for future use
            
        if not files:
            st.warning(f"No files have been uploaded yet in {category}.")
            return None

        # Format file names with upload datetime
        file_options = [
            f"{file['filename']}: {(datetime.fromisoformat(file['upload_datetime']) + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d')}"
            for file in files
        ]
        selected_option = st.sidebar.selectbox(f"Select a previously uploaded file ({len(file_options)} files)", file_options)

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
                st.error(
                    f"No file found with the name '{selected_filename}' in {category}. Please try again."
                )
                return None

    return uploaded_file
