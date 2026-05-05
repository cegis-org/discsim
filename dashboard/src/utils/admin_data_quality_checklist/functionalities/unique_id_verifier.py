import os
import streamlit as st
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.utils.utility_functions import read_uploaded_file,callAPI
import time

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

FIND_UNIQUE_IDS_ENDPOINT = f"{API_BASE_URL}/find_unique_ids"

@st.cache_data
def customCss():
    customcss = """
        <style>
        .stHorizontalBlock .stColumn:nth-child(2){
            display:flex;
            align-items: flex-end;
        }
        .st-key-functioncall button{
            background-color:#3b8e51;
            color:#fff;
            border:none;
        }
        .st-key-functioncall button:hover,
        .st-key-functioncall button:active,
        .st-key-functioncall button:focus,
        .st-key-functioncall button:focus:not(:active){
            color:#fff!important;
            border:none;
        }
        </style>
    """
    st.markdown(customcss, unsafe_allow_html=True)

def unique_id_verifier(uploaded_file):
    customCss()
    title_info_markdown = """
        Use this feature to let the system identify the list of unique IDs in the dataset.
        - Numerical columns, or combinations which are comprised of more numerical columns, will be given precedence while displaying the output.
        - If you have also used any of the other modules before, you can also use the same dataset used there by clicking the "Use existing dataset" button below.
        - Valid input format for dataset: xlsx or csv
        - A minimum of ONE column has to be selected
        - Max no. of selectable columns: As many as the number of column headers
    """
    col1,col2 = st.columns(2)
    col1.markdown("<h2 style='text-align: center;font-weight:800;color:#136a9a;'>Identify Unique ID(s)</h2>", unsafe_allow_html=True, help=title_info_markdown)
    st.markdown("<p style='color:#3b8e51;margin-bottom:20px'>This function helps you identify all the column(s) that are unique in the uploaded dataset, alongside their datatypes. The function outputs a list of unique ID(s) as a table that is downloadable</p>", unsafe_allow_html=True)

    if col2.button("Find Unique ID(s)",key="functioncall"):
        with st.spinner("Finding unique IDs..."):
            try:
                # total_start_time = time.perf_counter()
                file_details, filename, file_read_time = read_uploaded_file(uploaded_file)
                response , api_call_time = callAPI(file_details,filename,FIND_UNIQUE_IDS_ENDPOINT)
                # dataframe_start_time = time.perf_counter()
                if response.status_code == 200:
                    unique_ids = response.json()
                    
                    if unique_ids:
                        st.success("Unique IDs found!")
                        st.write("The best possible Column(s) or Combinations that can act as Unique ID are as follows:")
                        df = pd.DataFrame(unique_ids)
                        df['UniqueID'] = df['UniqueID'].apply(lambda x: ' + '.join(x))
                        df = df.rename(columns={'UniqueID': 'Unique ID (data type)', 'Numeric_DataTypes': 'Is Numeric'})
                        df.index.name = 'SN'
                        df.index = df.index + 1
                        st.dataframe(df['Unique ID (data type)'], use_container_width=True,key="uniquedatatable")

                    else:
                        st.warning("No unique identifiers found. All columns or combinations have at least one duplicate.")

                    # dataframe_end_time = time.perf_counter()  - dataframe_start_time
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    st.write("Response content:", response.content)
                    st.write("Response headers:", response.headers)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Traceback:", traceback.format_exc())

            # total_end_time = time.perf_counter()
                        
            # st.info("**Performance Metrics:**")
            # st.write(f"- File Reading: {file_read_time:.3f} seconds")
            # st.write(f"- API Response Time - Server Side: {api_call_time:.3f} seconds")
            # st.write(f"- DataFrame Processing - Client Side: {(dataframe_end_time):.3f} seconds")
            # st.write(f"- Total Execution: {(total_end_time - total_start_time):.3f} seconds")