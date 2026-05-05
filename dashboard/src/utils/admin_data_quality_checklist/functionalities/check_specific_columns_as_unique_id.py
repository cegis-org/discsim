import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import time
from src.utils.utility_functions import callAPIWithParam

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

UNIQUE_ID_CHECK_ENDPOINT = f"{API_BASE_URL}/unique_id_check"
@st.cache_data
def customCss():
    customcss = """
        <style>
        button[data-testid="stBaseButton-secondaryFormSubmit"], 
        .st-key-dropentryBtn button, 
        .st-key-verifyIDBtn button, 
        .st-key-create_uid_btn button{
            background-color:#3b8e51;
            color:#fff;
            border:none;
        }
        button[data-testid="stBaseButton-secondaryFormSubmit"]:hover,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:active,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:focus,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:focus:not(:active),
        .st-key-dropentryBtn button:hover,
        .st-key-dropentryBtn button:active,
        .st-key-dropentryBtn button:focus,
        .st-key-dropentryBtn button:focus:not(:active),
        .st-key-verifyIDBtn button:hover,
        .st-key-verifyIDBtn button:active,
        .st-key-verifyIDBtn button:focus,
        .st-key-verifyIDBtn button:focus:not(:active),
        .st-key-create_uid_btn button:hover,
        .st-key-create_uid_btn button:active,
        .st-key-create_uid_btn button:focus,
        .st-key-create_uid_btn button:focus:not(:active){
            color:#fff!important;
            border:none;
        }
        .st-key-uidCol label p::after, 
        .st-key-unique_col_name label p::after, 
        .st-key-unique_col_delimiter label p::after{ 
            content: " *";
            color: red;
        }
        </style>
    """
    st.markdown(customcss, unsafe_allow_html=True)

#Updates the session before relaoding the page -> helps to redirect page
def handle_click(newSelection):
    st.session_state.option_selection = newSelection

@st.fragment
def unique_id_creation_fragment(df, columns):
    with st.expander("Create New Unique ID Column from Selected Columns", expanded=True):
        new_col_name = st.text_input(
            "Enter name for new Unique ID column", 
            value="Unique_ID", 
            key="unique_col_name"
        )
        delimiter = st.text_input(
            "Enter delimiter to join columns", 
            value="_", 
            key="unique_col_delimiter"
        )
        if st.button("Create Unique ID Column", key="create_uid_btn"):
            if not new_col_name.strip():
                st.error("Column name cannot be empty.")
            else:
                if new_col_name in df.columns:
                    df.drop(columns=[new_col_name], inplace=True)
                else:
                    with st.spinner("Generating unique ID column..."):
                        try:
                            df[new_col_name] = df[columns].astype(str).agg(delimiter.join, axis=1)
                            st.success(f"New column '{new_col_name}' added to the dataset.")
                            st.info(f"Download this new set and upload again to check the frequency table to generate visuals")
                            df.index.name = 'SN'
                            df.index = df.index + 1
                            st.dataframe(df, use_container_width=True, hide_index=False)
                        except Exception as e:
                            st.error(f"Failed to create unique ID column: {str(e)}")

def check_specific_columns_as_unique_id(df):
    customCss()
    title_info_markdown = """
        Use this feature to check whether the column(s) you think form the unique ID is indeed the unique ID.
        - Verifies if selected column(s) can serve as a unique identifier for the dataset.
        - You can select up to 4 columns to check.
        - The function will return whether the selected column(s) can work as a unique ID.
        - Valid input format: CSV file
        - A minimum of ONE column must be selected.
    """
    st.markdown("<h2 style='text-align: center;font-weight:800;color:#136a9a;margin-top:-15px'>Verify Unique ID(s)</h2>", unsafe_allow_html=True, help=title_info_markdown)
    st.markdown("<p style='color:#3b8e51;margin-bottom:20px'>This function helps you verify if the column(s) that you selected serve(s) as a unique ID for your dataset. If you expect some column(s) to be a unique ID in your dataset, this function helps you verify the same.</p>", unsafe_allow_html=True)
    
    with st.form("unique_id_verification_form"):
        columns = st.multiselect(
            "Select column(s) to verify [up to 4]",
            max_selections=4,
            options=[str(col) for col in df.columns.tolist()],
            key="uidCol"
        )
        submit = st.form_submit_button("Check Unique ID")
    
    if columns and submit:
        # total_start_time = time.perf_counter()
        with st.spinner(f"Checking if {', '.join(columns)} form a unique ID..."):

            # file_read_start = time.perf_counter()
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
            # file_read_time = time.perf_counter() - file_read_start

            data = df_clean.where(pd.notnull(df_clean), None).to_dict('records')
            payload = {"data": data, "columns": columns}

            response, api_call_time = callAPIWithParam(payload, UNIQUE_ID_CHECK_ENDPOINT)
            
            # dataframe_start_time = time.perf_counter()
            if response.status_code == 200:
                try:
                    result = response.json()['result']
                except ValueError:
                    st.error("Received invalid JSON response from the server.")
                    return
                
                if result[1]:
                    #st.success("Yes, the selected column(s) can work as a Unique ID")
                    st.success(result[0])
                    unique_id_creation_fragment(df, columns)
                else:
                    #st.error("No, the select column(s) cannot work as a Unique ID")
                    st.error(result[0])

                    with st.expander("Show/Export the duplicates"):
                        duplicate_rows = df[df.duplicated(subset=columns, keep=False)]
                        duplicate_rows.index.name = 'SN'
                        duplicate_rows.index = duplicate_rows.index + 1
                        st.dataframe(duplicate_rows, use_container_width=True, hide_index=False)

                    paraField, colBtn = st.columns([2,1])
                    paraField.write("In case you want to drop/export the duplicate entries from the column(s), you can use the Inspect Duplicate Entries function")
                    dropentry = "Inspect Duplicate Entries"
                    colBtn.button(dropentry, on_click=handle_click, args=[dropentry],key="dropentryBtn")

                    paraField.write("In case you want to identify the Unique ID(s) in your file, you can use the Identify Unique ID(s) function")
                    verifyID = "Identify Unique ID(s)"
                    colBtn.button(verifyID, on_click=handle_click, args=[verifyID],key="verifyIDBtn")
            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"Error: {response.status_code} - {error_detail}")
            # dataframe_end_time = time.perf_counter()

            # total_end_time = time.perf_counter()
            # st.info("**Performance Metrics:**")
            # st.write(f"- Data Cleaning & Prep: {file_read_time:.3f} seconds")
            # st.write(f"- API Response Time (Server): {api_call_time:.3f} seconds")
            # st.write(f"- DataFrame Handling (Client): {(dataframe_end_time - dataframe_start_time):.3f} seconds")
            # st.write(f"- Total Execution Time: {(total_end_time - total_start_time):.3f} seconds")