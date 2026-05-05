import requests
import os
import io
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

def fetch_file_from_api(file_id):
    get_file_endpoint = f"{API_BASE_URL}/get_file/{file_id}"
    response = requests.get(get_file_endpoint)
    if response.status_code == 200:
        file_data = response.json()
        file_content = file_data["content"]
        # Convert the file content back to a file-like object
        uploaded_file = io.StringIO(file_content)
        uploaded_file.name = file_data["filename"]
        return uploaded_file
    else:
        raise Exception(f"Failed to fetch file from API: {response.text}")
