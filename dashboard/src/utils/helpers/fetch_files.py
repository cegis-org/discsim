import streamlit as st
import requests
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

GET_ALL_FILES_ENDPOINT = f"{API_BASE_URL}/list_files"
GET_FILE_ENDPOINT = f"{API_BASE_URL}/get_file"

# @st.cache_data(ttl=300)
def fetch_files_from_api(category):
    params = {"category": category}
    response = requests.get(f"{GET_ALL_FILES_ENDPOINT}", params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to retrieve file list. Status code: {response.status_code}")
        return []

@st.cache_data(ttl=300)
def fetch_file_from_api(file_id):
    file_response = requests.get(f"{GET_FILE_ENDPOINT}/{file_id}")
    if file_response.status_code == 200:
        file_data = file_response.json()
        file_content = file_data["content"]
        if file_content.startswith("\ufeff"):
            file_content = file_content.lstrip("\ufeff")
        
        uploaded_file = BytesIO(file_content.encode("utf-8"))  
        uploaded_file.name = file_data["filename"]
        return uploaded_file
    else:
        try:
            error_data = file_response.json()
            error_message = error_data.get("detail", "Unknown error occurred.")
        except Exception:
            error_message = file_response.text  # fallback if not JSON
        st.error(f"Failed to fetch file with ID {file_id}: {error_message}")
        return None