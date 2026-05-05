import json
import os
from src.utils.admin_data_quality_checklist.helpers.graph_functions import plot_pie_chart
import streamlit as st
import pandas as pd
import requests
import time
from dotenv import load_dotenv
from src.utils.utility_functions import read_uploaded_file,callAPIWithFileParam,fetch_dataframe

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

DROP_EXPORT_DUPLICATES_ENDPOINT = f"{API_BASE_URL}/drop_export_duplicates"
GET_PROCESSED_DATA_ENDPOINT = f"{API_BASE_URL}/get_processed_data"
GET_DATAFRAME_ENDPOINT = f"{API_BASE_URL}/get_dataframe"

def handle_click(newSelection):
    st.session_state.option_selection = newSelection

@st.cache_data
def customCss():
    customcss = """
        <style>
        button[data-testid="stBaseButton-secondaryFormSubmit"]{
            background-color:#3b8e51;
            color:#fff;
            border:none;
        }
        button[data-testid="stBaseButton-secondaryFormSubmit"]:hover,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:active,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:focus,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:focus:not(:active)
        {
            color:#fff!important;
            border:none;
        }
        .st-key-uidCol label p::after,
        .st-key-duplicateKeep label p::after { 
            content: " *";
            color: red;
        }
        </style>
    """
    st.markdown(customcss, unsafe_allow_html=True)
def drop_export_duplicate_entries(uploaded_file, df):
    customCss()
    st.session_state.drop_export_rows_complete = False
    title_info_markdown = """
        The function identifies duplicate entries in the dataset and returns the unique and the duplicate DataFrames individually.
        - Removes duplicate rows based on specified unique identifier column(s).
        - Options:
        - Select which duplicate to keep: first, last, or none.
        - Export duplicates to a separate file.
        - Supports processing large files in chunks for better performance.
        - Handles infinity and NaN values by replacing them with NaN.
        - Valid input format: CSV file
    """
    st.markdown("<h2 style='text-align: center;font-weight:800;color:#136a9a;margin-top:-15px'>Inspect Duplicate Entries</h2>", unsafe_allow_html=True, help=title_info_markdown)
    st.markdown("<p style='color:#3b8e51;margin-bottom:20px'>The function helps you to inspect if any duplicate entries are present in the unique identifier column(s) you selected. You can get a modified dataset with unique entries only</p>", unsafe_allow_html=True)

    with st.form('duplicate_entries'):
        col1, col2, col3 = st.columns(3)
        with col1:
            uid_col = st.multiselect("Select unique identifier column(s) [up to 4 columns]", df.columns.tolist(),key="uidCol",max_selections=4)
        with col2: 
            helpKeep = """
                1. First occurrence (in case duplicate, i.e. >1, entries exist, keeps the first occurrence and drops the rest)
                2. Last occurrence (in case duplicate, i.e. >1, entries exist, keeps the last occurrence and drops the rest)
                3. Drop all occurrences (in case duplicate, i.e. >1, entries exist, drops every occurrence
            """
            kept_row = st.selectbox("Which duplicate(s) to keep?", ["first", "last", "none"], help=helpKeep ,key="duplicateKeep")
        with col3:
            chunksize = st.number_input("Chunksize (optional, select 0 for no chunking)", min_value=0, step=1000, help="Use only for very large datasets, set to 10000 to start with")
        submit = st.form_submit_button("Check for duplicate entries")

    
    if submit:
        if not uid_col:
            st.error("Please select column")
            return 
        else:
            # total_start_time = time.perf_counter()
            with st.spinner("Processing..."):
                try:
                    file_bytes, filename, file_read_time = read_uploaded_file(uploaded_file)

                    payload = {
                        "uidCol": uid_col,
                        "keptRow": kept_row,
                        "chunksize": chunksize if chunksize > 0 else None
                    }

                    response, api_call_end = callAPIWithFileParam(file_bytes,payload,DROP_EXPORT_DUPLICATES_ENDPOINT)

                    if response.status_code == 200:
                        # dataframe_start = time.perf_counter()
                        result = response.json()
                        st.session_state.drop_export_entries_complete = True

                        # api_call_start_1 = time.perf_counter()
                        unique_df = fetch_dataframe('unique',GET_DATAFRAME_ENDPOINT)
                        # api_call_end_1 = time.perf_counter() - api_call_start_1

                        # api_call_start_2 = time.perf_counter()
                        duplicate_df = fetch_dataframe('duplicate',GET_DATAFRAME_ENDPOINT)
                        # api_call_end_2 = time.perf_counter() - api_call_start_2

                        # Visualize the results
                        unique_rows = len(unique_df)
                        duplicate_rows = len(duplicate_df)

                        fig = plot_pie_chart([f"Unique Entries", f"Duplicate Entries"], [unique_rows, duplicate_rows], "Dataset Composition")
                        st.plotly_chart(fig)
                        col5,col4 = st.columns(2)
                        # Display dataframes

                        col5.subheader("Unique Entries")
                        col5.text("In case you want to continue your analysis using the modified dataset, you can view and download the modified dataset with only unique entries here")
                        with col5.expander("Show/export unique entries"):
                            try:
                                st.write("")
                                paraField, colBtn = st.columns([2,1])
                                paraField.write("To further deep-dive into this data, download the file, upload it to the module, and use the Generate Frequency Table function")
                                dropentry = "Generate frequency table"
                                colBtn.button(dropentry, on_click=handle_click, args=[dropentry],key="dropentryBtn")
                                st.write("")
                                st.write("")

                                unique_df.index.name = 'SN'
                                unique_df.index = unique_df.index + 1
                                st.dataframe(unique_df, hide_index=True)
                            except Exception as e:
                                st.error(f"Error displaying unique rows: {str(e)}")

                        col4.subheader("Duplicate Entries")
                        col4.text("In case you want to use them later for your reference, you can view or download the duplicate rows dropped from the original dataset here")

                        if len(duplicate_df)>0:
                            with col4.expander("Show/export duplicate entries"):
                                st.write("")
                                paraField, colBtn = st.columns([2,1])
                                paraField.write("To further deep-dive into this data, download the file, upload it to the module, and use the Generate Frequency Table function")
                                dropentry = "Generate frequency table"
                                colBtn.button(dropentry, on_click=handle_click, args=[dropentry],key="dropentryBtns")
                                st.write("")
                                st.write("")
                                try:
                                    duplicate_df.index.name = 'SN'
                                    duplicate_df.index = duplicate_df.index + 1
                                    st.dataframe(duplicate_df, hide_index=False)
                                except Exception as e:
                                    st.error(f"Error displaying duplicate rows: {str(e)}")
                        else:
                            col4.warning("No Duplicate Entries")
                        # dataframe_end = time.perf_counter() - dataframe_start
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")

                    # total_end_time = time.perf_counter()

                    # st.info("**Performance Metrics:**")
                    # st.write(f"- File Reading: {(file_read_time):.3f} seconds")
                    # st.write(f"- API Response Time (Server): {(api_call_end + api_call_end_1 + api_call_end_2):.3f} seconds")
                    # st.write(f"- DataFrame Processing (Client): {(dataframe_end):.3f} seconds")
                    # st.write(f"- Total Execution Time: {(total_end_time - total_start_time):.3f} seconds")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

           

        