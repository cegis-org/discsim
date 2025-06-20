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

ZERO_ENTRIES_ENDPOINT = f"{API_BASE_URL}/zero_entries"

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
        .st-key-uidCol label p::after,
        .st-key-duplicateKeep label p::after { 
            content: " *";
            color: red;
        }
        </style>
    """
    st.markdown(customcss, unsafe_allow_html=True)
def zero_entries_analysis(uploaded_file, df):
    customCss()
    st.session_state.drop_export_rows_complete = False
    st.session_state.drop_export_entries_complete = False
    title_info_markdown = """
        The function returns the count and percentage of zero values for a variable, with optional filtering and grouping by a categorical variable.
        - Analyzes zero entries in a specified column of the dataset.
        - Options:
        - Select a column to analyze
        - Optionally group by a categorical variable
        - Optionally filter by a categorical variable
        - Provides the count and percentage of zero entries.
        - Displays a table of rows with zero entries.
        - Valid input format: CSV file
    """
    st.markdown("<h2 style='text-align: center;font-weight:800;color:#136a9a;margin-top:-15px'>Analyse Zero Entries</h2>", unsafe_allow_html=True, help=title_info_markdown)
    st.markdown("<p style='color:#3b8e51;margin-bottom:20px'>The function helps you analyse the zero entries present in your data. Furthermore, you can get a break down of the prevalence of zero entries among different groups of your data, or analyse the zero entries for only specific subset(s) of your data</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("")
        st.write("")
        st.write("")
        column_to_analyze = st.selectbox("Select a column you want to analyse for zero entries", df.select_dtypes(include='number').columns.tolist(),key="uidCol")
    with col2:
        group_by = st.selectbox("Do you want to break this down for particular groups of data? Please choose a (cateogorical) variable from your dataset", ["None"] + df.columns.tolist(), help="Analyze zero entries within distinct categories of another column. This is useful if you want to understand how zero values are distributed across different groups.")
    with col3:
        filter_by_col = st.selectbox("Do you want to restrict this analysis to a particular subset of data? Please choose the specific indicator and value for which you need this analysis", ["None"] + df.columns.tolist(), help="Focus on a specific subset of your data by selecting a specific value in another column. This is helpful when you want to analyze zero entries for a specific condition.")

    col4, col5, col6 = st.columns(3)
    if filter_by_col != "None":
        with col4:
            filter_by_value = st.selectbox("Choose the any value for which you need the analysis", df[filter_by_col].unique().tolist(),key="duplicateKeep")
        with col5:
            st.write("")
        with col6:
            st.write("")
        
    if st.button("Analyze Zero Entries",key="processBtn"):
        # total_start_time = time.perf_counter()
        with st.spinner("Analyzing zero entries..."):
            try:

                file_bytes, filename, file_read_time = read_uploaded_file(uploaded_file)

                payload = {
                    "column_to_analyze": column_to_analyze,
                    "group_by": group_by if group_by != "None" else None,
                    "filter_by": {filter_by_col: filter_by_value} if filter_by_col != "None" else None
                }

                response, api_call_end = callAPIWithFileParam(file_bytes,payload,ZERO_ENTRIES_ENDPOINT)

                if response.status_code == 200:
                    # dataframe_start = time.perf_counter()
                    result = response.json()

                    st.success(f"Zero entries analysed for column: '{column_to_analyze}'")
                    
                    if result["grouped"]:
                        a,b = st.columns(2)
                        a.metric(f"Total number of rows analysed",format(result['total_rows'],',d'),border=True)
                        b.metric(f"Zero entries",format(result['zero_entries'],',d'),border=True)
                        st.info(f"Results are grouped by column : {group_by}")
                        group_column_name = group_by  # Use the selected group-by column name
                        grouped_data = [
                            {
                                group_column_name: group_by,
                                "Zero Count": format(count, ',d'),
                                "Zero Percentage": f"{percentage:.2f}%",
                                "Total Rows": format(total, ',d')
                            }
                            for group_by, (count, percentage, total) in result["analysis"].items()
                        ]
                        grouped_df = pd.DataFrame(grouped_data)
                        
                        data = pd.DataFrame([(group, percentage, 100-percentage) for group, (count, percentage, totalrows) in result["analysis"].items()], columns=[group_column_name, 'Zero Entries', 'Non-Zero Entries'])
                        data = data.sort_values('Zero Entries', ascending=False)
                        fig = px.bar(data, x=group_column_name, y=['Zero Entries', 'Non-Zero Entries'], 
                                    title=f"Zero vs Non-Zero Entries by {group_column_name}",
                                    labels={'value': 'Percentage', 'variable': 'Entry Type'},
                                    color_discrete_map={'Zero Entries': '#9e2f17', 'Non-Zero Entries': '#3b8e51'})
                        fig.update_layout(barmode='relative', yaxis_title='Percentage',margin=dict(l=0, r=0, t=30, b=0),title_x=0.4)
                        fig.update_traces(texttemplate='%{y:.1f}%')
                        st.plotly_chart(fig)

                        with st.expander("Show tabular view"):
                            grouped_df = grouped_df.sort_values("Zero Count", ascending=False)
                            grouped_df.index.name = 'SN'
                            grouped_df.index = grouped_df.index + 1
                            st.dataframe(grouped_df, use_container_width=True, hide_index=False)

                    else:
                        count, percentage, total = result["analysis"]
                        a,b = st.columns(2)
                        a.metric(f"Total number of rows analysed",format(total,',d'),border=True)
                        b.metric(f"Zero entries",format(count,',d')+f"({percentage:.2f}%)",border=True)
                        
                        labels = ['Zero Entries', 'Non-Zero Entries']
                        values = [percentage, 100-percentage]
                        color_map = {
                                label: "#3b8e51" if "Non-Zero Entries" in label else "#9e2f17"
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
                    
                    if result["filtered"]:
                        st.info(f"Results are filtered by {filter_by_col} = {filter_by_value}")
                    
                    # Display the table of zero entries
                    if "zero_entries_table" in result:
                        zero_entries_df = pd.DataFrame(result["zero_entries_table"])
                        if column_to_analyze in zero_entries_df.columns:
                            zero_entries_df = zero_entries_df.sort_values(column_to_analyze, ascending=False)
                            with st.expander("Show/export rows with zero entries:"):

                                st.write("")
                                paraField, colBtn = st.columns([3,1])
                                paraField.write("To further deep-dive into this data, download the file, upload it to the module, and use the Generate Frequency Table function")
                                dropentry = "Generate frequency table"
                                colBtn.button(dropentry, on_click=handle_click, args=[dropentry],key="dropentryBtns")
                                st.write("")
                                st.write("")

                                zero_entries_df.index.name = 'SN'
                                zero_entries_df.index = zero_entries_df.index + 1
                                st.dataframe(zero_entries_df, use_container_width=True, hide_index=False)

                            #each instance group by
                            if group_by is not "None":
                                unique_values = zero_entries_df[group_by].unique()
                                st.write(f"### Splitting data by `{group_by}`")
                                for val in unique_values:
                                    with st.expander(f"Zero Entries for **{val}**"):
                                        st.dataframe(zero_entries_df[zero_entries_df[group_by] == val])

                        else:
                            st.warning(f"Zero entries not found.")

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
