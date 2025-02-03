import streamlit as st
import requests
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

GET_FILE_ENDPOINT = f"{API_BASE_URL}/get_file"

@st.cache_data
def fetch_file_from_api(file_id):
    file_response = requests.get(f"{GET_FILE_ENDPOINT}/{file_id}")
    if file_response.status_code == 200:
        file_data = file_response.json()
        file_content = file_data["content"].encode('utf-8')
        uploaded_file = BytesIO(file_content)
        uploaded_file.name = file_data["filename"]
        return uploaded_file
    else:
        st.error(f"Failed to fetch file with ID {file_id}.")
        return None
    
def get_file(file_id):
    # Check if the file is already in session state
    if 'uploaded_file' in st.session_state and st.session_state.get('file_id') == file_id:
        # Use the file from session state if it's already there
        return st.session_state.uploaded_file
    
    # Fetch the file using the cached fetch function
    uploaded_file = fetch_file_from_api(file_id)

    # If file is fetched successfully, store it in session state for future use
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.file_id = file_id  # Store the file ID to check later
    
    return uploaded_file