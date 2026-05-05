import json
import os
import traceback
import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv
import plotly.express as px
import time
from src.utils.utility_functions import read_uploaded_file,callAPIWithFileParam

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

FREQUENCY_TABLE_ENDPOINT = f"{API_BASE_URL}/frequency_table"

@st.cache_data
def customCss():
    customcss = """
        <style>
        .st-key-processBtn button{
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
        .st-key-uidCol label p::after,
        .st-key-duplicateKeep label p::after,
        .st-key-duplicateValue label p::after { 
            content: " *";
            color: red;
        }
        </style>
    """
    st.markdown(customcss, unsafe_allow_html=True)

def frequency_table_analysis(uploaded_file, df):
    customCss()
    title_info_markdown = '''
        This function takes a variable as a user input and returns the frequency table of number and share of observations of each unique value present in the variable.
        - Generates a frequency table for a specified column in the dataset.
        - Options:
        - Select a column to analyze
        - Specify the number of top frequent values to display separately
        - Optionally group by a categorical variable
        - Optionally filter by a categorical variable
        - Provides counts and percentages for each unique value in the selected column.
        - Valid input format: CSV file
    '''
    st.markdown("<h2 style='text-align: center;font-weight:800;color:#136a9a;margin-top:-15px'>Generate frequency table</h2>", unsafe_allow_html=True, help=title_info_markdown)
    st.markdown("<p style='color:#3b8e51;margin-bottom:20px'>The function helps you generate frequency tables from the dataset you have uploaded. If you want breakdowns of data by categorical variables of your choice, this function is helpful</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        column_to_analyze = st.selectbox("Select the categorical variable for which you want to generate the frequency table", df.columns.tolist(),key="uidCol")
    with col2:
        st.write("")
        top_n = st.selectbox(
            "Do you want to order your frequency table?",["None","Ascending","Descending"]
        )
    with col3:
        group_by = st.selectbox("Do you want to further break your analysis down by another categorical variable?", ["None"] + df.columns.tolist(), help="Analyze missing entries within distinct categories of another column. This is useful if you want to understand how missing values are distributed across different groups.")
    
    col4, col5, col6 = st.columns(3)
    with col4:    
        filter_by_col = st.selectbox("Do you want to restrict the analysis to a subset of your data?", ["None"] + df.columns.tolist(), help="Focus on a specific subset of your data by selecting a specific value in another column. This is helpful when you want to analyze missing entries for a specific condition.")
    with col5:
        st.write("")
    with col6:
        st.write("")
    if filter_by_col != "None":
        filter_by_value = col5.selectbox("Enter value for which you want to restrict the analysis", df[filter_by_col].unique().tolist(),key="duplicateValue")

    if st.button("Generate Frequency Table",key="processBtn"):
        # total_start_time = time.perf_counter()
        with st.spinner("Generating frequency table..."):
            try:

                file_bytes, filename, file_read_time = read_uploaded_file(uploaded_file)

                payload = {
                    "column_to_analyze": column_to_analyze,
                    "top_n": top_n,
                    "group_by": group_by if group_by != "None" else None,
                    "filter_by": {filter_by_col: filter_by_value} if filter_by_col != "None" else None,
                }

                response, api_call_end = callAPIWithFileParam(file_bytes,payload,FREQUENCY_TABLE_ENDPOINT)

                if response.status_code == 200:
                    dataframe_start = time.perf_counter()
                    result = response.json()

                    st.metric(f"Total number of rows analysed",format(result['total'],',d'),border=True)

                    if result["filtered"]:
                        st.info(f"Results are filtered by {filter_by_col} = {filter_by_value}")
                    
                    top_n = top_n.lower()
                    if result["grouped"]:
                        st.info(f"Combined Frequency Table for All Groups: {group_by}")
                        full_table, top_n_table = result["analysis"]

                        if top_n in ["ascending", "descending"]:
                            st.info(f"Frequency table sorted by {top_n.capitalize()} frequency:")
                            display_df = pd.DataFrame(top_n_table)
                        else:
                            st.info("Full frequency table (unsorted):")
                            display_df = pd.DataFrame(full_table)

                        fig = px.bar(
                            display_df,
                            x="count",
                            y=column_to_analyze,
                            color=group_by,
                            orientation="h",
                            text=display_df["count"].apply(lambda x: f"{x:,}"), 
                            barmode="group",
                            color_discrete_sequence= ["#006898", "#bd5942", "#9e2f17", "#3b8e51", "#3390B3", "#66AFC6"]
                        )
                        fig.update_traces(width=0.2)
                        fig.update_layout(
                            barcornerradius=5,
                            xaxis_tickformat=",",
                            legend_title=group_by,
                            bargap=0.1,
                            height=400,
                            margin=dict(t=0, b=0),
                            xaxis=dict(title=None,showgrid=False,showticklabels=True),
                            yaxis=dict(title=None,showgrid=False,showticklabels=True,autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)

                        display_df = display_df[[column_to_analyze, group_by, "count", "share %"]]
                        display_df.rename(columns={
                            "count": "Frequency",
                            "share %": "Share %"
                        }, inplace=True)
                        display_df["Share %"] = display_df["Share %"].round(1).astype(str) + " %"
                        display_df["Frequency"] = display_df["Frequency"].apply(lambda x: f"{x:,}")
                        display_df.index.name = 'SN'
                        display_df.index = display_df.index + 1

                        with st.expander("Show/Export Data:"):
                            st.dataframe(display_df, use_container_width=True, hide_index=False)

                    else:
                        st.info(f"Frequency by {column_to_analyze}")
                        full_table, ordered_table = result["analysis"]

                        if top_n in ["ascending", "descending"]:
                            st.info(f"Frequency table sorted by {top_n.capitalize()} frequency:")
                            display_df = pd.DataFrame(ordered_table)
                        else:
                            st.info("Full frequency table (unsorted):")
                            display_df = pd.DataFrame(full_table)
                        display_df["Frequency"] = display_df["Frequency"].apply(lambda x: f"{x:,}")

                        # Use sorted or full table depending on your logic
                        chart_df = pd.DataFrame(ordered_table if top_n in ["ascending", "descending"] else full_table)
                        chart_df.columns = [column_to_analyze, "Frequency", "share %"]
                        chart_df["count_display"] = chart_df["Frequency"].apply(lambda x: f"{x:,}")

                        # Create horizontal bar chart
                        fig = px.bar(
                            chart_df,
                            x="Frequency",
                            y=column_to_analyze,
                            orientation="h",
                            text="count_display",
                            color_discrete_sequence=["#006898"]
                        )
                        fig.update_traces(width=0.5)
                        fig.update_layout(
                            barcornerradius=5,
                            xaxis_tickformat=",",
                            margin=dict(t=0, b=0),
                            height=300,
                            bargroupgap=0.1,
                            xaxis=dict(title=None,showgrid=False,showticklabels=True),
                            yaxis=dict(title=None,showgrid=False,showticklabels=True,autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)

                        # Format and show the table
                        display_df.columns = [column_to_analyze, "Frequency", "Share %"]
                        display_df["Share %"]=display_df["Share %"].round(1).astype(str)+' %'
                        display_df.index.name = "SN"
                        display_df.index = display_df.index + 1

                        with st.expander("Show/Export Data:"):
                            st.dataframe(display_df, use_container_width=True, hide_index=False)

                    # dataframe_end = time.perf_counter() - dataframe_start
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Traceback:", traceback.format_exc())

            # total_end_time = time.perf_counter()

            # st.info("**Performance Metrics:**")
            # st.write(f"- File Reading: {(file_read_time):.3f} seconds")
            # st.write(f"- API Response Time (Server): {(api_call_end):.3f} seconds")
            # st.write(f"- DataFrame Processing (Client): {(dataframe_end):.3f} seconds")
            # st.write(f"- Total Execution Time: {(total_end_time - total_start_time):.3f} seconds")
