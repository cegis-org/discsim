import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import plotly.io as pio
import base64

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
POST_SURVEY_ANALYSIS_ENDPOINT = f"{API_BASE_URL}/post_survey_analysis"

def execute_post_survey_analysis(uploaded_file, df):
    st.header("ECD Discrepancy Analysis")

    # Display the dataframe
    with st.expander("Uploaded Data Head:"):
        st.dataframe(df.head(), use_container_width=True, hide_index=True)

    # Check if necessary columns are present
    required_columns = [
        'child', 'L0_height', 'L1_height', 'L0_weight', 'L1_weight', 
        'L0_id', 'L1_id', 'L0_name', 'L1_name', 
        'wasting_L0', 'stunting_L0', 'underweight_L0', 
        'wasting_L1', 'stunting_L1', 'underweight_L1'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
        return

    # Inputs for margins of error (Side by Side)
    col1, col2 = st.columns(2)
    with col1:
        margin_of_error_height = st.number_input(
            "Acceptable Margin for Height (cm)", 
            min_value=0.0, 
            value=0.0, 
            step=0.1,
            help="Acceptable margin of error for height measurements."
        )
    with col2:
        margin_of_error_weight = st.number_input(
            "Acceptable Margin for Weight (kg)", 
            min_value=0.0, 
            value=0.0, 
            step=0.1,
            help="Acceptable margin of error for weight measurements."
        )

    # Prepare data for API
    uploaded_file.seek(0)
    files = {"file": ("uploaded_file.csv", uploaded_file, "text/csv")}
    data = {
        "margin_of_error_height": margin_of_error_height,
        "margin_of_error_weight": margin_of_error_weight
    }

    # Send request to API
    with st.spinner("Calculating discrepancy scores..."):
        response = requests.post(POST_SURVEY_ANALYSIS_ENDPOINT, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        grouped_scores = result.get('grouped_discrepancy_scores', [])
        plots = result.get('plots', {})

        if not grouped_scores:
            st.info("No discrepancy scores calculated. Check your data and margin of error.")
            return

        # Convert to DataFrame for better handling
        scores_df = pd.DataFrame(grouped_scores)
        # Sort the dataframe by composite discrepancy score in descending order
        scores_df = scores_df.sort_values('composite_discrepancy_score', ascending=False)

        with st.expander("Discrepancy Measures per L0 and L1:"):
            st.dataframe(scores_df, use_container_width=True, hide_index=True)

        # Display Composite Discrepancy Scores
        composite_scores = scores_df[['L0_name', 'L1_name', 'composite_discrepancy_score']]
        with st.expander("Composite Discrepancy Scores:"):
            st.dataframe(composite_scores, use_container_width=True, hide_index=True)

        def display_plot(plot_json, caption):
            try:
                fig = pio.from_json(plot_json)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load plot: {caption}. Error: {str(e)}")

        # Display Plots
        st.subheader("Discrepancy in Measurement")

        # Arrange the average height and weight discrepancy plots side by side
        col3, col4 = st.columns(2)
        with col3:
            # Plot 1: Average Height Discrepancy (cm) vs L0
            if 'height_discrepancy_plot' in plots:
                display_plot(plots['height_discrepancy_plot'], "Average Height Discrepancy (cm) per L0")
            else:
                st.warning("Height Discrepancy Plot not available.")
        with col4:
            # Plot 2: Average Weight Discrepancy (kg) vs L0
            if 'weight_discrepancy_plot' in plots:
                display_plot(plots['weight_discrepancy_plot'], "Average Weight Discrepancy (kg) per L0")
            else:
                st.warning("Weight Discrepancy Plot not available.")
                
        # Combined average discrepancy plot
        if 'combined_discrepancy_plot' in plots:
            display_plot(plots['combined_discrepancy_plot'], "Combined Average Discrepancy per L0")
        else:
            st.warning("Combined Discrepancy Plot not available.")

        # Arrange the height and weight measurement accuracy plots side by side
        col5, col6 = st.columns(2)
        st.subheader("Discrepancy in Percentage")
        with col5:
            # Plot 3: Height Measurement Accuracy (%) vs L0
            if 'height_accuracy_plot' in plots:
                display_plot(plots['height_accuracy_plot'], "Height Measurement Accuracy (%) per L0")
            else:
                st.warning("Height Measurement Accuracy Plot not available.")
        with col6:
            # Plot 4: Weight Measurement Accuracy (%) vs L0
            if 'weight_accuracy_plot' in plots:
                display_plot(plots['weight_accuracy_plot'], "Weight Measurement Accuracy (%) per L0")
            else:
                st.warning("Weight Measurement Accuracy Plot not available.")

        # Arrange the three classification accuracy plots side by side
        col7, col8 = st.columns(2)
        with col7:
            # Plot 5: Classification Accuracy - Wasting vs L0
            if 'classification_wasting_plot' in plots:
                display_plot(plots['classification_wasting_plot'], "Classification Accuracy - Wasting vs L0")
            else:
                st.warning("Classification Accuracy - Wasting Plot not available.")
        with col8:
            # Plot 6: Classification Accuracy - Stunting vs L0
            if 'classification_stunting_plot' in plots:
                display_plot(plots['classification_stunting_plot'], "Classification Accuracy - Stunting vs L0")
            else:
                st.warning("Classification Accuracy - Stunting Plot not available.")
        # with col9:
        #     # Plot 7: Classification Accuracy - Underweight vs L0
        #     if 'classification_underweight_plot' in plots:
        #         display_plot(plots['classification_underweight_plot'], "Classification Accuracy - Underweight vs L0")
        #     else:
        #         st.warning("Classification Accuracy - Underweight Plot not available.")
        
        # # Plot the composite discrepancy scores
        # st.subheader("Composite Discrepancy Score Plot")

        # if 'composite_discrepancy_plot' in plots:
        #     display_plot(plots['composite_discrepancy_plot'], "Composite Discrepancy Score per L0")
        # else:
        #     st.warning("Composite Discrepancy Score Plot not available.")

        # Optionally, provide download link for discrepancy scores
        # st.subheader("Download Discrepancy Scores")
        # csv = scores_df.to_csv(index=False)
        # b64 = base64.b64encode(csv.encode()).decode()  # some strings
        # href = f'<a href="data:file/csv;base64,{b64}" download="discrepancy_scores.csv">Download CSV File</a>'
        # st.markdown(href, unsafe_allow_html=True)

    else:
        # Attempt to extract error message from response
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text
        st.error(f"Error in calculation: {error_detail}")
