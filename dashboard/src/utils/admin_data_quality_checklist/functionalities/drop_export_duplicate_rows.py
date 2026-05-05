import json
import os
from src.utils.admin_data_quality_checklist.helpers.graph_functions import plot_pie_chart
import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv
import time
from src.utils.utility_functions import read_uploaded_file,callAPI,fetch_dataframe


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

DROP_EXPORT_DUPLICATE_ROWS_ENDPOINT = f"{API_BASE_URL}/drop_export_duplicate_rows"
GET_PROCESSED_DATA_ENDPOINT = f"{API_BASE_URL}/get_processed_data"
GET_DATAFRAME_ENDPOINT = f"{API_BASE_URL}/get_dataframe"

def handle_click(newSelection):
    st.session_state.option_selection = newSelection

@st.cache_data
def customCss():
    customcss = """
        <style>
        .st-key-processBtn button,.st-key-dropentryBtn button, .st-key-dropentryBtns button{
            background-color:#3b8e51;
            color:#fff;
            border:none;
        }
        .st-key-processBtn button:hover,
        .st-key-processBtn button:active,
        .st-key-processBtn button:focus,
        .st-key-processBtn button:focus:not(:active){
            color:#fff!important;
            border:none;
        }
        </style>
    """
    st.markdown(customcss, unsafe_allow_html=True)


def drop_export_duplicate_rows(uploaded_file):
    customCss()
    st.session_state.drop_export_entries_complete = False
    title_info_markdown = """
        This function checks for fully duplicate rows in the dataset and returns the unique and the duplicate DataFrames individually.
        - Analyzes the dataset to find completely duplicated rows.
        - Removes duplicate rows based on all columns.
        - Options:
        - Select which duplicate to keep: first, last, or none.
        - Export duplicates to a separate file.
        - Provides the count and percentage of duplicate rows in the dataset.
        - Valid input format: CSV file
    """
    col1,col2 = st.columns(2)
    col1.markdown("<h2 style='text-align: center;font-weight:800;color:#136a9a;margin-top:-15px;'>Inspect Duplicate Rows</h2>", unsafe_allow_html=True, help=title_info_markdown)
    st.markdown("<p style='color:#3b8e51;margin-bottom:20px'>The function helps you to inspect if any duplicate rows exist in the dataset. You can get a modified dataset with unique rows only</p>", unsafe_allow_html=True)

    if col2.button("Check for duplicate rows",key="processBtn"):
        # total_start_time = time.perf_counter()
        with st.spinner("Processing..."):
            try:

                file_bytes, filename, file_read_time = read_uploaded_file(uploaded_file)

                response, api_call_time = callAPI(file_bytes, filename, DROP_EXPORT_DUPLICATE_ROWS_ENDPOINT)

                if response.status_code == 200:
                    # dataframe_start = time.perf_counter()
                    result = response.json()
                    st.session_state.drop_export_rows_complete = True

                    # api_call_start_1 = time.perf_counter()
                    unique_df = fetch_dataframe('unique',GET_DATAFRAME_ENDPOINT)
                    # api_call_end_1 = time.perf_counter() - api_call_start_1

                    # api_call_start_2 = time.perf_counter()
                    duplicate_df = fetch_dataframe('duplicate',GET_DATAFRAME_ENDPOINT)
                    # api_call_end_2 = time.perf_counter() - api_call_start_2
                    
                    # Visualize the results
                    unique_rows = len(unique_df)
                    duplicate_rows = len(duplicate_df)

                    fig = plot_pie_chart([f"Unique Rows", f"Duplicate Rows"], [unique_rows, duplicate_rows], "Dataset Composition")
                    st.plotly_chart(fig)
                    
                    # Display dataframes
                    st.subheader("Unique Rows")
                    with st.expander("Unique Rows:"):

                        st.write("")
                        paraField, colBtn = st.columns([3,1])
                        paraField.write("To further deep-dive into this data, download the file, upload it to the module, and use the Generate Frequency Table function")
                        dropentry = "Generate frequency table"
                        colBtn.button(dropentry, on_click=handle_click, args=[dropentry],key="dropentryBtn")
                        st.write("")
                        st.write("")

                        unique_df.index.name = 'SN'
                        unique_df.index = unique_df.index + 1
                        st.dataframe(unique_df, hide_index=False)

                    if len(duplicate_df)>0:
                        st.subheader("Duplicate Rows")
                        with st.expander("Duplicate Rows:"):

                            st.write("")
                            paraField, colBtn = st.columns([3,1])
                            paraField.write("To further deep-dive into this data, download the file, upload it to the module, and use the Generate Frequency Table function")
                            dropentry = "Generate frequency table"
                            colBtn.button(dropentry, on_click=handle_click, args=[dropentry],key="dropentryBtns")
                            st.write("")
                            st.write("")

                            duplicate_df.index.name = 'SN'
                            duplicate_df.index = duplicate_df.index + 1
                            st.dataframe(duplicate_df, hide_index=False)
                    # dataframe_end = time.perf_counter() - dataframe_start
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

            # total_end_time = time.perf_counter()

            # st.info("**Performance Metrics:**")
            # st.write(f"- File Reading: {(file_read_time):.3f} seconds")
            # st.write(f"- API Response Time (Server): {(api_call_time + api_call_end_1 + api_call_end_2):.3f} seconds")
            # st.write(f"- DataFrame Processing (Client): {(dataframe_end):.3f} seconds")
            # st.write(f"- Total Execution Time: {(total_end_time - total_start_time):.3f} seconds")