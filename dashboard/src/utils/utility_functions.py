import streamlit as st
import os
from PIL import Image
from streamlit_navigation_bar import st_navbar
from dotenv import load_dotenv
load_dotenv()


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
logo_path = os.path.join(root_dir, "logo.jpg")
im = Image.open(logo_path)

logo_page_path = os.path.join(root_dir, "logo_white.png")
im_page = Image.open(logo_page_path)

def set_page_config(sideBar = 'expanded'):
    st.logo(im_page)

    st.set_page_config(
        page_title="DiscSim | CEGIS",
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
            "max-width": "30rem",
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
    return st_navbar(["Pre Survey", "Admin Data Quality", "Post Survey"],selected=SelectedNav,styles=navStyles,options=navOptions) # type: ignore

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