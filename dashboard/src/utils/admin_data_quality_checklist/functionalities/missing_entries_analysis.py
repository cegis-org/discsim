import json
import os
import traceback
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from dotenv import load_dotenv
import time
from src.utils.utility_functions import read_uploaded_file,callAPIWithFileParam


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

MISSING_ENTRIES_ENDPOINT = f"{API_BASE_URL}/missing_entries"

def handle_click(newSelection):
    st.session_state.option_selection = newSelection

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
        .st-key-uidCol label p::after,.st-key-duplicateKeep label p::after { 
            content: " *";
            color: red;
        }
        </style>
    """
    st.markdown(customcss, unsafe_allow_html=True)
def missing_entries_analysis(uploaded_file, df):
    customCss()
    st.session_state.drop_export_rows_complete = False
    st.session_state.drop_export_entries_complete = False    
    title_info_markdown = """
        This function returns the count and percentage of missing values for a given variable, with optional filtering and grouping by a categorical variable.
        - Analyzes the dataset to find missing entries in a specified column.
        - Optionally groups or filters the analysis by other categorical columns.
        - Provides a table of rows with missing entries.
        - Valid input format: CSV file
    """
    st.markdown("<h2 style='text-align: center;font-weight:800;color:#136a9a;margin-top:-15px'>Analyse Missing Entries</h2>", unsafe_allow_html=True, help=title_info_markdown)
    st.markdown("<p style='color:#3b8e51;margin-bottom:20px'>The function helps you analyse the missing entries present in your data. Furthermore, you can get a break down of the prevalence of missing entries among different groups of your data, or analyse the missing entries for only specific subset(s) of your data</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("")
        st.write("")
        st.write("")
        column_to_analyze = st.selectbox("Select a column you want to analyse for missing entries", options=df.columns.tolist(), index=0,key="uidCol")
    with col2:
        group_by = st.selectbox("Do you want to break this down for particular groups of data? Please choose a (cateogorical) variable from your dataset", options=["None"] + df.columns.tolist(), index=0, help="Analyze missing entries within distinct categories of another column. This is useful if you want to understand how missing values are distributed across different groups.")
    with col3:
        filter_by_col = st.selectbox("Do you want to restrict this analysis to a particular subset of data? Please choose the specific indicator and value for which you need this analysis", options=["None"] + df.columns.tolist(), index=0, help="Focus on a specific subset of your data by selecting a specific value in another column. This is helpful when you want to analyze missing entries for a specific condition.")
    
    col4, col5, col6 = st.columns(3)
    if filter_by_col != "None":
        with col4:
            filter_by_value = st.selectbox("Choose the any value for which you need the analysis", df[filter_by_col].unique().tolist(),key="duplicateKeep")
        with col5:
            st.write("")
        with col6:
            st.write("")
    else:
        filter_by_value = None

    # Analyze Missing Entries
    if st.button("Analyze Missing Entries",key="processBtn"):
        # total_start_time = time.perf_counter()
        with st.spinner("Analyzing missing entries..."):
            try:
                file_bytes, filename, file_read_time = read_uploaded_file(uploaded_file)

                payload = {
                    "column_to_analyze": column_to_analyze,
                    "group_by": group_by if group_by != "None" else None,
                    "filter_by": {filter_by_col: filter_by_value} if filter_by_col != "None" else None
                }

                response, api_call_end = callAPIWithFileParam(file_bytes,payload,MISSING_ENTRIES_ENDPOINT)
                                                    
                if response.status_code == 200:
                    # dataframe_start = time.perf_counter()
                    result = response.json()
                    st.success(f"Missing entries analysed for column: '{column_to_analyze}'")

                    if result["grouped"]:
                        if result["filtered"] == False:
                            a,b = st.columns(2)
                            a.metric(f"Total number of rows analysed",format(result['total_rows'],',d'),border=True)
                            b.metric(f"Missing entries",format(result['zero_entries'],',d'),border=True)
                        st.info(f"Results are grouped by column : {group_by}")
                        grouped_data = []
                        for group, (count, percentage, total) in result["analysis"].items():
                            grouped_data.append({
                                group_by: group, 
                                "Total Rows": format(total,',d'),
                                "Missing Entries": format(count,',d'),
                                "Missing Percentage": f"{percentage:.2f}%" if percentage is not None else "N/A"
                            })
                        if result["filtered"]:
                            a,b = st.columns(2)
                            a.metric(f"Total number of rows analysed",format(total,',d'),border=True)
                            b.metric(f"Missing entries",format(count,',d')+f"({percentage:.2f}%)",border=True)

                        grouped_df = pd.DataFrame(grouped_data)
                        grouped_df = grouped_df.sort_values("Missing Entries", ascending=False)
                        
                        # Create a 100% stacked column chart
                        data = pd.DataFrame([(group, percentage, 100-percentage if percentage is not None else 0) 
                                            for group, (count, percentage ,total) in result["analysis"].items()],
                                            columns=[group_by, 'Missing', 'Recorded'])
                        data = data.sort_values('Missing', ascending=False)
                        fig = px.bar(data, x=group_by, y=['Missing', 'Recorded'], 
                                    title=f"Missing vs Recorded Entries by {group_by}",
                                    labels={'value': 'Percentage', 'variable': 'Status'},
                                    color_discrete_map={'Missing': '#9e2f17', 'Recorded': '#3b8e51'},
                                    text='value')
                        fig.update_layout(barmode='relative', yaxis_title='Percentage',margin=dict(l=0, r=0, t=30, b=0),title_x=0.4)
                        fig.update_traces(texttemplate='%{text:.1f}%')
                        st.plotly_chart(fig)

                        # Center-align just Missing Count and Missing Percentage
                        with st.expander("Show tabular view"):
                            grouped_df.index.name = 'SN'
                            grouped_df.index = grouped_df.index + 1
                            st.dataframe(grouped_df, use_container_width=True, hide_index=False)
                        
                    else:
                        count, percentage, total = result["analysis"]
                        if percentage is not None:
                            a,b = st.columns(2)
                            a.metric(f"Total number of rows analysed",format(total,',d'),border=True)
                            b.metric(f"Missing entries",format(count,',d')+f"({percentage:.2f}%)",border=True)
                            labels = ['Missing Entries', 'Recorded Entries']
                            values = [percentage, 100-percentage]
                            color_map = {
                                label: "#3b8e51" if "Recorded Entries" in label else "#9e2f17"
                                for label in labels
                            }
                            fig = px.pie(
                                names=labels, 
                                values=values, 
                                color=labels,
                                color_discrete_map=color_map)
                            fig.update_layout(
                                margin=dict(l=0, r=0, t=0, b=0),
                                height=400,
                            )
                            fig.update_traces(textinfo='percent+label')
                            st.plotly_chart(fig)
                        else:
                            st.write(f"Missing entries: {count} (percentage unavailable)")
                    
                    if result["filtered"]:
                        st.info(f"Results are filtered by {filter_by_col} = {filter_by_value}")
                        
                    # Display the table of missing entries
                    if "missing_entries_table" in result:   
                        if not result["missing_entries_table"]:
                            st.warning("The missing entries table is empty.")
                        else:
                            missing_entries_df = pd.DataFrame(result["missing_entries_table"])
                                                                    
                            if column_to_analyze in missing_entries_df.columns:
                                missing_entries_df = missing_entries_df.sort_values(column_to_analyze, ascending=False)
                                
                            else:
                                st.warning(f"Column '{column_to_analyze}' not found in the missing entries table. Displaying unsorted data.")
                            
                            with st.expander("Show/export rows with missing entries"):

                                st.write("")
                                paraField, colBtn = st.columns([3,1])
                                paraField.write("To further deep-dive into this data, download the file, upload it to the module, and use the Generate Frequency Table function")
                                dropentry = "Generate frequency table"
                                colBtn.button(dropentry, on_click=handle_click, args=[dropentry],key="dropentryBtns")
                                st.write("")
                                st.write("")
                            
                                missing_entries_df.index.name = 'SN'
                                missing_entries_df.index = missing_entries_df.index + 1
                                st.dataframe(missing_entries_df, use_container_width=True, hide_index=False)

                            #each instance group by
                            if group_by is not "None":
                                unique_values = missing_entries_df[group_by].unique()
                                st.write(f"### Splitting data by `{group_by}`")
                                for val in unique_values:
                                    with st.expander(f"Missing Entries for **{val}**"):
                                        st.dataframe(missing_entries_df[missing_entries_df[group_by] == val])

                    else:
                        st.error("The 'missing_entries_table' key is not present in the API response.")

                    # dataframe_end = time.perf_counter() - dataframe_start
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {response.status_code} - {error_detail}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Traceback:", traceback.format_exc())

            # total_end_time = time.perf_counter()

            # st.info("**Performance Metrics:**")
            # st.write(f"- File Reading: {(file_read_time):.3f} seconds")
            # st.write(f"- API Response Time (Server): {(api_call_end):.3f} seconds")
            # st.write(f"- DataFrame Processing (Client): {(dataframe_end):.3f} seconds")
            # st.write(f"- Total Execution Time: {(total_end_time - total_start_time):.3f} seconds")
