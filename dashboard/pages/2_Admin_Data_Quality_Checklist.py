import streamlit as st
import pandas as pd
from src.utils.state_management import initialize_states, reset_session_states, reset_upload
from src.utils.helpers.file_upload import handle_file_upload
from src.utils.admin_data_quality_checklist.helpers.functionality_map import execute_functionality, sidebar_functionality_select
from src.utils.utility_functions import set_page_config,setFooter,setheader
from src.utils.admin_data_quality_checklist.helpers.about_the_data import abouthepage
import streamlit.components.v1 as components
set_page_config()

def admin_data_quality_check():
    notecss = """
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div.st-key-notefordata){
            padding:5px;
        }
        </style>
    """
    st.markdown(notecss, unsafe_allow_html=True)

    # File selection
    file_option = st.sidebar.radio("Choose an option:", ("Upload a file", "Select a previously uploaded file"))

    # Initialize states
    initialize_states()

    # Clear relevant session state when switching options
    if st.session_state.previous_file_option != file_option:
        st.session_state.uploaded_file = None
        st.session_state.uploaded_file_id = None
        reset_session_states()
        st.session_state.previous_file_option = file_option

    uploaded_file = handle_file_upload(file_option, category="admin_data_quality_checklist")

    collapse_sidebar = False


    if uploaded_file is not None:
        collapse_sidebar = True

        if collapse_sidebar:
            collapse_sidebar_js = """
            <script>
            setTimeout(function() {
                const btn = window.parent.document.querySelector('[data-testid="stBaseButton-headerNoPadding"]');
                if (window.innerWidth < 768) {
                    if (btn) {
                        btn.click();
                    }
                }
            }, 500);
            </script>
            """
            components.html(collapse_sidebar_js, height=0)

        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_file = uploaded_file
        if uploaded_file != st.session_state.previous_uploaded_file:
            reset_session_states()
            st.session_state.previous_uploaded_file = uploaded_file
        st.sidebar.divider()

        with st.sidebar.container(border=True,key="notefordata"):
            st.markdown("<h5>Unselect the function to preview the data",unsafe_allow_html=True)

        functionality = sidebar_functionality_select()
        if(functionality):
            st.session_state.navbar_selection = functionality
            execute_functionality(functionality, uploaded_file, df)
        else:
            abouthepage(uploaded_file)

    else:
        st.info("Please upload a CSV file to begin.")
        reset_session_states()
        st.session_state.previous_uploaded_file = None

if st.session_state.get('reset_upload', False):
    reset_upload()
    st.rerun()

if __name__ == "__main__":
    selectedNav = setheader("Admin Data Diagnostic")
    if selectedNav == "Home":
            st.switch_page("Home.py")
    if selectedNav == "Intervention Design":
            st.switch_page("pages/1_Intervention_Design.py")
    if selectedNav == "Intervention Analytics":
            st.switch_page("pages/3_Intervention_Analytics.py")
    admin_data_quality_check()

    setFooter()