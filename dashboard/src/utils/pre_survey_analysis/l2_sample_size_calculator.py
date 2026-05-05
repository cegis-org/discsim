import streamlit as st
import requests
import os
import plotly.graph_objects as go
from dotenv import load_dotenv
from src.utils.pre_survey_analysis.error_handling import check_errors

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

L2_SAMPLE_SIZE_ENDPOINT = f"{API_BASE_URL}/l2-sample-size"


def l2_sample_size_calculator():
    st.markdown("<h2 style='text-align: center;'>L2 Sample Size Calculator", unsafe_allow_html=True)
        
    # Input fields
    total_samples = st.number_input("Total number of samples", help="The total number of data points that third party will sample (typically between 100-1000). Range > 0", min_value=1, value=100)
    average_truth_score = st.slider("Average truth score", help="The expected average truth score across all blocks (typically between 0.2-0.5). Ideally, should be based on some real data from the sector in question. Higher is worse (i.e. more mismatches between subordinate and 3P). 0 < Range < 1", min_value=0.0, max_value=1.0, value=0.5)
    variance_across_blocks = st.slider("Variance across blocks", help="The expected standard deviation of mean truth score across blocks (typically between 0.1-0.5). Ideally, should be based on some real data from the sector in question. The higher this value, the easier it will be to correctly rank the blocks. Range > 0", min_value=0.0, max_value=1.0, value=0.1)
    variance_within_block = st.slider("Variance within block", help="The expected standard deviation across subordinates within a block (typically between 0.1-0.5). Ideally, should be based on some real data from the sector in question. The higher this value, the more difficult it will be to correctly rank the blocks. Range > 0", min_value=0.0, max_value=1.0, value=0.1)
    level_test = st.selectbox("Level of test", ["Block", "District", "State"], help="The aggregation level at which 3P will test and give reward/punishment.")
    n_subs_per_block = st.number_input("Number of subordinates per block", help="The number of subordinates in a block. Range > 1", min_value=1, value=10)
    n_blocks_per_district = st.number_input("Number of blocks per district", help="The number of blocks in a district. Range >= 1", min_value=1, value=5)
    n_district = st.number_input("Number of districts", help="Number of districts. Range >= 1", min_value=1, value=1)
    n_simulations = st.number_input("Number of simulations", help="By default, this should be set to 100. The number of times the algorithm will be run to estimate the number of samples required. Higher n_simulations will give a more accurate answer, but will take longer to run. Range > 1", min_value=1, value=100)
    min_sub_per_block = st.number_input("Minimum subordinates per block", help="Minimum number of subordinates to be measured in each block. By default, this should be set to 1. 0 < Range < n_sub_per_block", min_value=1, value=1)

    if st.button("Calculate L2 Sample Size"):
        input_data = {
            "total_samples": total_samples,
            "average_truth_score": average_truth_score,
            "variance_across_blocks": variance_across_blocks,
            "variance_within_block": variance_within_block,
            "level_test": level_test,
            "n_subs_per_block": n_subs_per_block,
            "n_blocks_per_district": n_blocks_per_district,
            "n_district": n_district,
            "n_simulations": n_simulations,
            "min_sub_per_block": min_sub_per_block
        }
            
        # Error handling
        error_status, error_message = check_errors(input_data)
        if error_status == 0:
            st.error(f"Error: {error_message}")
            return

        response = requests.post(L2_SAMPLE_SIZE_ENDPOINT, json=input_data)
            
        if response.status_code == 200:
            result = response.json()
            st.success(f"L2 Sample Size: {result['value']['n_samples']}")
            st.info(result['message'])
                
            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(result['value']['true_disc']))), y=result['value']['true_disc'], mode='lines', name='True Discrepancy'))
            fig.add_trace(go.Scatter(x=list(range(len(result['value']['meas_disc']))), y=result['value']['meas_disc'], mode='markers', name='Measured Discrepancy'))
            fig.update_layout(
                title="True vs Measured Discrepancy",
                xaxis_title=f"{level_test} Index",
                yaxis_title="Discrepancy Score"
            )
            st.plotly_chart(fig)
        else:
            st.error(f"Error: {response.json()['detail']}")