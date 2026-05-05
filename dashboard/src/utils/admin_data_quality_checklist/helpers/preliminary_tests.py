import streamlit as st
import requests
import os
from dotenv import load_dotenv
from src.utils.admin_data_quality_checklist.helpers.functionality_map import get_relevant_functionality
from src.utils.admin_data_quality_checklist.helpers.display_preview import display_data_preview

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

PRELIMINARY_TESTS_ENDPOINT = f"{API_BASE_URL}/preliminary_tests"

def run_preliminary_tests(uploaded_file):
    if 'preliminary_test_result' in st.session_state and uploaded_file == st.session_state.previous_uploaded_file:
        # If the same file is uploaded, simply return the stored results
        with st.expander("Preliminary tests on the data file uploaded",expanded=True):
            st.warning("Warnings")
            for warning in st.session_state.preliminary_test_result_response["warnings"]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"- {warning}")
                with col2:
                    relevant_func = get_relevant_functionality(warning)
                    if st.button(f"Check {relevant_func}", key=f"warning_button_{warning}"):
                        st.session_state.navbar_selection = relevant_func
            display_data_preview(st.session_state.previous_uploaded_file)
        return st.session_state.preliminary_test_result  # Return cached result
    with st.spinner("Running preliminary tests on the uploaded file..."):
        with st.expander("Preliminary tests on the data file uploaded",expanded=True):
            try:
                uploaded_file.seek(0)  # Reset file pointer before reading again
                files = {"file": ("uploaded_file.csv", uploaded_file, "text/csv")}
                response = requests.post(PRELIMINARY_TESTS_ENDPOINT, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.preliminary_test_result_response = result
                    if result["status"] == 0:
                        if result["warnings"]:
                            st.warning("Warnings")
                            for warning in result["warnings"]:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"- {warning}")
                                with col2:
                                    relevant_func = get_relevant_functionality(warning)
                                    if st.button(f"Check {relevant_func}", key=f"warning_button_{warning}"):
                                        st.session_state.navbar_selection = relevant_func
                            # Display data preview
                            display_data_preview(uploaded_file)
                        # Store the test result in session state for future use
                        st.session_state.preliminary_test_result = True
                        st.session_state.previous_uploaded_file = uploaded_file  # Save the file to session state
                        return True
                    else:
                        st.error("Preliminary tests failed. Please check your file and try again.")
                        st.session_state.preliminary_test_result = False
                        st.session_state.previous_uploaded_file = uploaded_file
                        return False
                else:
                    st.error(f"Error in preliminary tests: {response.status_code}")
                    st.session_state.preliminary_test_result = False
                    st.session_state.previous_uploaded_file = uploaded_file
                    return False
            except Exception as e:
                st.error(f"Error during preliminary tests: {str(e)}")
                st.session_state.preliminary_test_result = False
                st.session_state.previous_uploaded_file = uploaded_file
                return False
