import json
import os
import traceback
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from dotenv import load_dotenv
from src.utils.admin_data_quality_checklist.helpers.graph_functions import plot_pie_chart

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

ZERO_ENTRIES_ENDPOINT = f"{API_BASE_URL}/zero_entries"

def zero_entries_analysis(uploaded_file, df):
    customcss = """
        <style>
        div[data-testid="stExpander"] summary{
            padding:0.4rem 1rem;
        }
        .stHorizontalBlock{
            //margin-top:-30px;
        }
        .st-key-processBtn button{
            background-color:#3b8e51;
            color:#fff;
            border:none;
        }
        .st-key-processBtn button:hover,.st-key-processBtn button:active,.st-key-processBtn button:focus,st-key-processBtn button:focus:not(:active){
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
        column_to_analyze = st.selectbox("Select a column you want to analyse for zero entries", df.select_dtypes(include='number').columns.tolist(),key="uidCol")
    with col2:
        group_by = st.selectbox("Do you want to break this down for particular groups of data? Please choose a (cateogorical) variable from your dataset (optional)", ["None"] + df.columns.tolist(), help="Analyze zero entries within distinct categories of another column. This is useful if you want to understand how zero values are distributed across different groups.")
    with col3:
        filter_by_col = st.selectbox("Do you want to restrict this analysis to a particular subset of data? Please choose the specific indicator and value for which you need this analysis (optional)", ["None"] + df.columns.tolist(), help="Focus on a specific subset of your data by selecting a specific value in another column. This is helpful when you want to analyze zero entries for a specific condition.")

    col4, col5, col6 = st.columns(3)
    if filter_by_col != "None":
        with col4:
            filter_by_value = st.selectbox("Choose the any value for which you need the analysis", df[filter_by_col].unique().tolist(),key="duplicateKeep")
        with col5:
            st.write("")
        with col6:
            st.write("")
        
    if st.button("Analyze Zero Entries",key="processBtn"):
        with st.spinner("Analyzing zero entries..."):
            try:
                uploaded_file.seek(0)  # Reset file pointer
                files = {"file": ("uploaded_file.csv", uploaded_file, "text/csv")}
                payload = {
                    "column_to_analyze": column_to_analyze,
                    "group_by": group_by if group_by != "None" else None,
                    "filter_by": {filter_by_col: filter_by_value} if filter_by_col != "None" else None
                }
                response = requests.post(
                    ZERO_ENTRIES_ENDPOINT,
                    files=files,
                    data={"input_data": json.dumps(payload)}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result["grouped"]:
                        st.write("Zero entries by group: " + group_by)
                        group_column_name = group_by  # Use the selected group-by column name
                        grouped_data = [{group_column_name: group, "Zero Count": count, "Zero Percentage": f"{percentage:.2f}%"}
                                        for group, (count, percentage) in result["analysis"].items()]
                        grouped_df = pd.DataFrame(grouped_data)
                        
                        data = pd.DataFrame([(group, percentage, 100-percentage) for group, (count, percentage) in result["analysis"].items()], columns=[group_column_name, 'Zero', 'Non-Zero'])
                        data = data.sort_values('Zero', ascending=False)
                        fig = px.bar(data, x=group_column_name, y=['Zero', 'Non-Zero'], 
                                    title=f"Zero vs Non-Zero Entries by {group_column_name}",
                                    labels={'value': 'Percentage', 'variable': 'Entry Type'},
                                    color_discrete_map={'Zero': '#9e2f17', 'Non-Zero': '#3b8e51'})
                        fig.update_layout(barmode='relative', yaxis_title='Percentage',showlegend=False,margin=dict(l=0, r=0, t=30, b=0),title_x=0.4)
                        fig.update_traces(texttemplate='%{y:.1f}%', textposition='inside')
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
                        #Show Dataframe when Zero Entries Exists
                        if count > 0:
                            analysis_df = pd.DataFrame([{"Zero Count": count, "Zero Percentage": f"{percentage:.2f}%"}])
                            analysis_df.index.name = 'SN'
                            analysis_df.index = analysis_df.index + 1
                            st.dataframe(analysis_df, use_container_width=True, hide_index=False)
                        
                        labels = ['Zero', 'Non-Zero']
                        values = [percentage, 100-percentage]
                        color_map = {
                                label: "#3b8e51" if "Non-Zero" in label else "#9e2f17"
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
                            showlegend= False,
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig)
                    
                    if result["filtered"]:
                        st.info(f"Results are filtered by {filter_by_col} = {filter_by_value}")
                    
                    # Display the table of zero entries
                    if "zero_entries_table" in result:
                        zero_entries_df = pd.DataFrame(result["zero_entries_table"])
                        if column_to_analyze in zero_entries_df.columns:
                            zero_entries_df = zero_entries_df.sort_values(column_to_analyze, ascending=False)
                            with st.expander("Rows with Zero Entries:"):
                                zero_entries_df.index.name = 'SN'
                                zero_entries_df.index = zero_entries_df.index + 1
                                st.dataframe(zero_entries_df, use_container_width=True, hide_index=False)
                        else:
                            st.warning(f"Zero entries not found.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Traceback:", traceback.format_exc())