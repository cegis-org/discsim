import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

ERROR_HANDLING_ENDPOINT = f"{API_BASE_URL}/error-handling"

def check_errors(params):
    response = requests.post(ERROR_HANDLING_ENDPOINT, json={"params": params})
    if response.status_code == 200:
        result = response.json()
        return result["status"], result["message"]
    else:
        return 0, f"Error in error handling: {response.json()['detail']}"
