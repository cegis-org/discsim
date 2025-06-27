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

INDICATOR_FILL_RATE_ENDPOINT = f"{API_BASE_URL}/indicator_fill_rate"

@st.cache_data
def is_numeric_column(series):
    return pd.api.types.is_numeric_dtype(series) or series.dtype == 'object' and series.str.isnumeric().all()

@st.cache_data
def is_string_column(series):
    return pd.api.types.is_string_dtype(series)

@st.cache_data
def is_datetime_column(series: pd.Series) -> bool:
    """
    Check if a pandas Series is of datetime type or can be interpreted as datetime.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # parsing first few non-null entries (for robustness)
    non_null_values = series.dropna().head(5)
    for val in non_null_values:
        try:
            pd.to_datetime(val)
            return True
        except (ValueError, TypeError):
            continue
    return False
@st.cache_data
def get_numeric_operations():
    return ['<', '<=', '>', '>=', '==', '!=', 'Between', 'Compare Equals', 'Compare Not Equals']
@st.cache_data
def get_string_operations():
    return ['Contains', 'Does not contain', 'Equals', 'Not equals', 'Compare Equals', 'Compare Not Equals']

@st.cache_data
def get_datetime_operations():
    return ['<', '<=', '>', '>=', '==', '!=', 'Between', 'Compare Equals', 'Compare Not Equals']

def display_detailed_data(data: dict, invalid_labels: list = [None], include_zero_as_separate_category_flag: bool = False):
    categories = ["missing", "valid"]
    if include_zero_as_separate_category_flag:
        categories.insert(1, "zero")
    for label in (invalid_labels):
        categories.insert(-1, label)
    for category in categories:
        entries = data.get(category, None)
        if entries is not None and len(entries) > 0:
            category_df = pd.DataFrame(entries)
            with st.expander(f"Show/Export {category.capitalize()} Entries:"):
                category_df.index.name = 'SN'
                category_df.index = category_df.index + 1
                st.dataframe(category_df, use_container_width=True, hide_index=False)
        else:
            st.warning(f"No {category.capitalize()} entries found.")

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

def indicator_fill_rate_analysis(uploaded_file, df):
    customCss()
    st.session_state.drop_export_rows_complete = False
    st.session_state.drop_export_entries_complete = False
    title_info_markdown = """
        This function analyzes a variable for missing, zero, and other invalid values, returning counts and percentages in a table format, with optional filtering or grouping by a categorical variable and customizable invalid value conditions (e.g., value > x).
        - Analyzes the fill rate and data quality of a specified column in the dataset.
        - Options:
        - Select a column to analyze
        - Optionally group by a categorical variable
        - Optionally filter by a categorical variable
        - Specify a custom condition for invalid values
        - Provides counts and percentages for missing, zero, invalid, and valid values.
        - Displays samples of missing, zero, invalid, and valid data.
        - Valid input format: CSV file
    """
    st.markdown("<h2 style='text-align: center;font-weight:800;color:#136a9a;margin-top:-15px'>Analyse the data quality of an indicator</h2>", unsafe_allow_html=True, help=title_info_markdown)
    st.markdown("<p style='color:#3b8e51;margin-bottom:20px'>The following function helps you assess the data quality of the indicator you choose. On choosing an indicator, you will be able to see the share of missing entries, zero entries, invalid entries, and the share of valid and usable entries. You have the option to define your own invalid criteria for numerical, string, and date-time indicators. A higher share of valid entries indicates good data quality</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("")
        column_to_analyze = st.selectbox("Select the indicator to analyse", df.columns.tolist(),key="uidCol")
    with col2:
        group_by = st.selectbox("Do you want to break down your analysis by any categorical variable? You can categorise the analysis", ["None"] + df.columns.tolist())
    with col3:
        filter_by_col = st.selectbox("Do you want to analyse only a subset of your data? You can filter your data", ["None"] + df.columns.tolist())
    
    col31, col32, col33 = st.columns(3)
    if filter_by_col != "None":
        with col31:
            filter_by_value = st.selectbox("Enter the value for which you want the analysis", df[filter_by_col].unique().tolist())
        with col32:
            st.write("")
        with col33:
            st.write("")
    
    num_conditions = st.number_input("How many invalid conditions? [add up to 3]", min_value=0, max_value=3, value=0, step=1)
    invalid_conditions = []
    include_zero_as_separate_category = False
    if is_numeric_column(df[column_to_analyze]):
        include_zero_as_separate_category = st.checkbox("Include zero entries as a separate category", value=True)
    if num_conditions:
        if is_numeric_column(df[column_to_analyze]):
            st.write("Define a criteria for invalid values")
            for i in range(num_conditions):
                st.markdown(f"**Condition {i+1}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    label = st.text_input(f"Criteria Name (spaces will be removed)", f"Invalid{i+1}", max_chars=15)
                with col2:
                    operation = st.selectbox(f"Operation", get_numeric_operations(), key=f"op{i}")
                if operation == "Between":
                    col4, col5 = col3.columns(2)
                    with col4:
                        lower = st.number_input(f"Lower bound (inclusive)", key=f"between_low_{i}")
                    with col5:
                        upper = st.number_input(f"Upper bound (inclusive)", key=f"between_high_{i}")
                    value = (lower, upper)
                elif operation in ["Compare Equals", "Compare Not Equals"]: 
                        valid_compare_cols = [col for col in df.columns if col != column_to_analyze]
                        value = col3.selectbox("Select column to compare with", valid_compare_cols, key=f"str_compare_col_{i}")
                else:
                    with col3:
                        value = st.number_input(f"Value", key=f"val{i}")

                invalid_conditions.append({
                    "label": label.strip().replace(" ", ""),
                    "operation": operation,
                    "value": value
                })
        elif is_string_column(df[column_to_analyze]):
            st.write("Set condition for invalid string values:")
            for i in range(num_conditions):
                st.markdown(f"**Condition {i+1}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    label = st.text_input(f"Criteria Name (spaces will be removed)", f"Invalid{i+1}", max_chars=15)
                with col2:
                    operation = st.selectbox(f"Operation", get_string_operations(), key=f"st{i}")
                with col3:
                    if operation in ["Contains", "Does not contain"]:
                        value = st.text_input(f"Enter value", key=f"str_val_text_{i}")
                    elif operation in ["Compare Equals", "Compare Not Equals"]: 
                        valid_compare_cols = [col for col in df.columns if col != column_to_analyze]
                        value = st.selectbox("Select column to compare with", valid_compare_cols, key=f"str_compare_col_{i}")
                    else:
                        value = st.selectbox(f"Select value", df[column_to_analyze].dropna().unique().tolist(), key=f"str_val_select_{i}")

                invalid_conditions.append({
                    "label": label.strip().replace(" ", ""),
                    "operation": operation,
                    "value": value
                })
            include_zero_as_separate_category = False
        elif is_datetime_column(df[column_to_analyze]):
            st.write("Set condition for invalid datetime values:")
            for i in range(num_conditions):
                st.markdown(f"**Condition {i+1}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    label = st.text_input(f"Criteria Name (spaces will be removed)", f"Invalid{i+1}", max_chars=15, key=f"dt_label_{i}")
                with col2:
                    operation = st.selectbox(f"Operation", get_datetime_operations(), key=f"dt_op_{i}")
                if operation == "Between":
                    col4, col5 = col3.columns(2)
                    with col4:
                        start_date = st.date_input(f"Start date (Exclusive)", key=f"dt_start_{i}")
                    with col5:
                        end_date = st.date_input(f"End date (Inclusive)", key=f"dt_end_{i}")
                    value = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                elif operation in ["Compare Equals", "Compare Not Equals"]: 
                        valid_compare_cols = [col for col in df.columns if col != column_to_analyze]
                        value = col3.selectbox("Select column to compare with", valid_compare_cols, key=f"date_compare_col_{i}")
                else:
                    with col3:
                        single_date = st.date_input(f"Select date", key=f"dt_value_{i}")
                        value = single_date.strftime('%Y-%m-%d')

                invalid_conditions.append({
                    "label": label.strip().replace(" ", ""),
                    "operation": operation,
                    "value": value
                })
            include_zero_as_separate_category = False
        else:
            st.write("The column should include either numbers, strings, or dates.")
            invalid_conditions = None

    if st.button("Analyse the indicator values",key="processBtn"):
        # total_start_time = time.perf_counter()
        with st.spinner("Analyzing indicator fill rate..."):
            try:

                file_bytes, filename, file_read_time = read_uploaded_file(uploaded_file)

                payload = {
                    "column_to_analyze": column_to_analyze,
                    "group_by": group_by if group_by != "None" else None,
                    "filter_by": {filter_by_col: filter_by_value} if filter_by_col != "None" else None,
                    "invalid_conditions": invalid_conditions,
                    "include_zero_as_separate_category": include_zero_as_separate_category
                }

                response, api_call_end = callAPIWithFileParam(file_bytes,payload,INDICATOR_FILL_RATE_ENDPOINT)
                
                if response.status_code == 200:
                    # dataframe_start = time.perf_counter()
                    try:
                        result = response.json()
                    except ValueError:
                        st.error("Failed to parse response from the server.")

                    invalid_labels = [cond["label"] for cond in invalid_conditions] if invalid_conditions else []
                    
                    if result["grouped"]:
                        st.info("Indicator Fill Rate Report by Group:")
                        st.metric(f"Total number of rows analysed",format(result['total'],',d'),border=True)

                        if result["filtered"]:
                            st.info(f"Results are filtered by {filter_by_col} = {filter_by_value}")

                        # Prepare data for 100% stacked column chart
                        all_groups_data = []
                        for group, analysis in result["analysis"].items():
                            analysis_df = pd.DataFrame(analysis)
                            analysis_df['Group'] = group
                            all_groups_data.append(analysis_df)
                        combined_df = pd.concat(all_groups_data, ignore_index=True)
                        # Calculate percentages within each group
                        combined_df['Percentage'] = combined_df.groupby('Group')['Number of observations'].transform(lambda x: x / x.sum() * 100).round(1).astype(str) + ' %'
                        color_map = {
                            "Valid": "#3b8e51",
                            "Missing": "#9e2f17",
                            "Zero": "#bd5942",
                        }
                        invalid_colors = ["#006898", "#3390B3", "#66AFC6", "#99C7D9", "#005272", "#003E4F", "#002B33", "#005E8A", "#007AA6", "#00507A"]
                        for i, cat in enumerate(invalid_labels):
                            if cat not in color_map:
                                color_map[cat] = invalid_colors[i % len(invalid_colors)]
                        # Create the 100% stacked column chart
                        fig = px.bar(combined_df, x='Group', y='Percentage', color='Category',color_discrete_map = color_map, labels={"x": group_by, 'value': "Percentage"}, barmode='relative',text='Percentage' ) 
                        fig.update_traces(textangle=0)
                        st.plotly_chart(fig)

                        # Display detailed data for each group
                        for group, analysis in result["analysis"].items():
                            st.subheader(f"Group: `{group}`")
                            analysis_df = pd.DataFrame(analysis)
                            analysis_df["Number of observations"] = analysis_df["Number of observations"].apply(lambda x: f"{int(x):,}")
                            analysis_df["Percentage of observations"]=analysis_df["Percentage of observations"].astype(str)+' %'
                            st.dataframe(analysis_df, use_container_width=True, hide_index=True)
                            display_detailed_data(result["detailed_data"][group],invalid_labels,include_zero_as_separate_category)
                            st.write("---")
                    else:
                        st.info("Indicator Fill Rate Result:")
                        st.metric(f"Total number of rows analysed",format(result['total'],',d'),border=True)

                        if result["filtered"]:
                            st.info(f"Results are filtered by {filter_by_col} = {filter_by_value}")

                        analysis_df = pd.DataFrame(result["analysis"])
                        # Create a simple pie chart of percentages
                        sorted_data = sorted(zip(analysis_df['Percentage of observations'], analysis_df['Category']), reverse=True)
                        sorted_values, sorted_labels = zip(*sorted_data)  

                        color_map = {
                            "Valid": "#3b8e51",
                            "Missing": "#9e2f17",
                            "Zero": "#bd5942",
                        }
                        invalid_colors = ["#006898", "#3390B3", "#66AFC6", "#99C7D9", "#005272", "#003E4F", "#002B33", "#005E8A", "#007AA6", "#00507A"]
                        for i, cat in enumerate(sorted_labels):
                            if cat not in color_map:
                                color_map[cat] = invalid_colors[i % len(invalid_colors)]

                        fig = px.pie(
                            names=sorted_labels, 
                            values=sorted_values, 
                            color=sorted_labels,
                            color_discrete_map=color_map)
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=40, b=0),
                            height=400,
                        )
                        fig.update_traces(textinfo='percent+label')
                        st.plotly_chart(fig)

                        with st.expander("Show/Export data:"):
                            analysis_df["Number of observations"] = analysis_df["Number of observations"].apply(lambda x: f"{int(x):,}")
                            analysis_df["Percentage of observations"]=analysis_df["Percentage of observations"].astype(str)+' %'
                            st.dataframe(analysis_df, use_container_width=True, hide_index=True)

                    
                        if invalid_conditions:
                            invalid_desc = ", ".join([f"{col['label']}: {col['operation']} {col['value']}" for col in invalid_conditions])
                            st.info(f"Custom invalid conditions applied to column `{column_to_analyze}`: {invalid_desc}")

                        display_detailed_data(result["detailed_data"],invalid_labels,include_zero_as_separate_category)
                    
                    # dataframe_end = time.perf_counter() - dataframe_start
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                # st.write("Traceback:", traceback.format_exc())

            # total_end_time = time.perf_counter()

            # st.info("**Performance Metrics:**")
            # st.write(f"- File Reading: {(file_read_time):.3f} seconds")
            # st.write(f"- API Response Time (Server): {(api_call_end):.3f} seconds")
            # st.write(f"- DataFrame Processing (Client): {(dataframe_end):.3f} seconds")
            # st.write(f"- Total Execution Time: {(total_end_time - total_start_time):.3f} seconds")