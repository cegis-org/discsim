import streamlit as st
import os
from PIL import Image
from streamlit_navigation_bar import st_navbar
from dotenv import load_dotenv
load_dotenv()
import requests
from urllib.parse import urlparse
import time
import json
import pandas as pd
import io


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
logo_path = os.path.join(root_dir, "logo.jpg")
im = Image.open(logo_path)

logo_page_path = os.path.join(root_dir, "logo_white.png")
im_page = Image.open(logo_page_path)

PUBLIC_URL = os.getenv("PUBLIC_URL")
def set_page_config(sideBar = 'expanded'):
    st.logo(im_page,link=PUBLIC_URL)

    st.set_page_config(
        page_title="VALIData | CEGIS",
        layout="wide",
        page_icon=im,
        initial_sidebar_state=sideBar,
    )
    loadcss(os.path.join(root_dir, "custom.css"))
    userAvatar()

def loadcss(file_path: str):
    try:
        with open(file_path) as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
    except Exception as e:
        st.error(f"An error occurred while loading CSS: {e}")

def userAvatar():
     if 'user_name' not in st.session_state:
          st.session_state['user_name'] = "CEGIS"
     @st.dialog("Update Username")
     def userDialog():
          userInput= st.text_input("Enter your name",st.session_state['user_name'])
          if st.button("Save"):
               st.session_state['user_name']=userInput
               st.rerun()

     with st.chat_message("human",avatar="https://avatar.iran.liara.run/public"):
          if st.button(st.session_state['user_name'] +  ':material/expand_more:'):
               userDialog()

def setheader(SelectedNav = None):
    navStyles = {
        "nav": {
            "background-color": "#136a9a",
            "justify-content": "center",
        },
        "div": {
            "max-width": "50rem",
        },
        "span": {
            "color": "#fff",
            "font-weight": "700",
            "padding": "14px",
        },
        "active": {
            "color": "#136a9a",
            "background-color":"#fff",
            "padding": "14px",
        },
     }
    navOptions = {
        "fix_shadow":False,
        "hide_nav":False
    }
    return st_navbar(["Home", "Intervention Design", "Admin Data Diagnostic", "Intervention Analytics"],selected=SelectedNav,styles=navStyles,options=navOptions) # type: ignore

def setFooter():
         # Footer using markdown with custom HTML
    footer = """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #136a9a;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: white;
            left:0;
            z-index:99999999;
        }
        .footer p{
            margin:0;
            font-size:14px;
        }
        </style>
        <div class="footer">
            <p> CEGIS © 2025 | All Rights Reserved.</p>
        </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

def clearAllSessions():
    for key in list(st.session_state.keys()):
        if key != "user_name":
            del st.session_state[key]
    st.success("Sessions cleared!")

def read_uploaded_file(uploaded_file):
    """Read uploaded file and return bytes and filename."""
    start_time = time.perf_counter()
    try:
        if not uploaded_file:
            raise ValueError("No file uploaded.")
        if not uploaded_file.name.lower().endswith(('.csv')):
            raise ValueError("Only CSV files are supported.")
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        if not file_bytes:
            raise ValueError("Uploaded file is empty.")
        file_io = io.BytesIO(file_bytes)
        end_time = time.perf_counter()
        return {"file": (uploaded_file.name, file_io, "text/csv")}, uploaded_file.name, end_time - start_time
    except Exception as e:
        st.error(f"Failed to read uploaded file: {str(e)}")
        raise

@st.cache_data(max_entries=10)
def callAPI(file_details: dict, filename: str, url: str):
    start_time = time.perf_counter()
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid API URL.")
        response = requests.post(url, files=file_details)
        response.raise_for_status()
        end_time = time.perf_counter()
        return response, end_time - start_time
    except requests.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        raise
    except ValueError as e:
        st.error(f"Invalid input: {str(e)}")
        raise

@st.cache_data(max_entries=10)
def callAPIWithParam(payload: dict, url: str):
    start_time = time.perf_counter()
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid API URL.")
        payload = payload
        response = requests.post(url, json=payload)
        response.raise_for_status()
        end_time = time.perf_counter()
        return response, end_time - start_time
    except requests.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        raise
    except ValueError as e:
        st.error(f"Invalid input: {str(e)}")
        raise

@st.cache_data(max_entries=10)
def callAPIWithFileParam(file_details: dict, payload: dict, url: str):
    start_time = time.perf_counter()
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid API URL")
        input_data = json.dumps(payload)
        response = requests.post(
            url,
            files=file_details,
            data={"input_data": input_data}
        )
        response.raise_for_status()
        end_time = time.perf_counter()
        return response, end_time - start_time
    except requests.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        raise
    except ValueError as e:
        st.error(f"Invalid input: {str(e)}")
        raise

@st.cache_data
def fetch_dataframe(data_type: str, url: str):
    return pd.DataFrame(requests.get(f"{url}?data_type={data_type}").json())