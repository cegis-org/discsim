import streamlit as st
import requests
import os
from dotenv import load_dotenv
from src.utils.pre_survey_analysis.error_handling import check_errors

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

L1_SAMPLE_SIZE_ENDPOINT = f"{API_BASE_URL}/l1-sample-size"

def l1_sample_size_calculator():
    st.markdown("<h2 style='text-align: center;'>L1 Sample Size Calculator", unsafe_allow_html=True)
    
    # Input fields
    min_n_samples = st.number_input("Minimum number of samples", help="The minimum number of data points that a supervisor will sample (typically 1). Range > 0", min_value=1, value=1)
    max_n_samples = st.number_input("Maximum number of samples",  help="The maximum number of data points that a supervisor can sample (if this is not high enough, the guarantee may not be possible and the algorithm will ask you to increase this). Range > min_n_samples", min_value=min_n_samples + 1, value=100)
    n_subs_per_block = st.number_input("Number of subordinates per block", help="The number of subordinates that one supervisor will test. Range > 0", min_value=1, value=10)
    n_blocks_per_district = st.number_input("Number of blocks per district", min_value=1, value=5)
    n_district = st.number_input("Number of districts", min_value=1, value=1)
    level_test = st.selectbox("Level of test", ["Block", "District", "State"])
    percent_punish = st.slider("Percentage of subordinates to be punished", help="The percentage of subordinates that will be punished. This should be less than 100% (the total number of subordinates). The higher this number, the easier it is to guarantee that worst offenders will be caught, so increase this if the number of samples being returned is too high. 0 < Range <= 100", min_value=0.0, max_value=100.0, value=10.0)
    percent_guarantee = st.slider("Percentage of worst offenders guaranteed", help="The percentage of worst offenders that we can guarantee will be present in the set of subordinates that are punished. The closer this number is to n_punish, the more difficult it is to guarantee, so decrease this if the number of samples being returned is too high.  0 < Range <= 100", min_value=0.0, max_value=percent_punish, value=5.0)
    confidence = st.slider("Confidence", help="The probability that n_guarantee worst offenders will be present in the set of n_punish subordinates with highest discrepancy scores. The higher this probability, the more difficult it is to guarantee, so decrease this if the number of samples being returned is too high. 0 < Range < 1", min_value=0.0, max_value=1.0, value=0.9)
    n_simulations = st.number_input("Number of simulations", help="By default, this should be set to 100. The number of times the algorithm will be run to estimate the number of samples required. Higher n_simulations will give a more accurate answer, but will take longer to run. Range > 1", min_value=1, value=100)
    min_disc = st.slider("Minimum discrepancy score", help="Minimum discrepancy score to be used for simulation. By default, set to 0 (no discrepancy between subordinate and supervisor). If you are working with a sector in which you have reason to believe the lowest observed discrepancy scores are higher than 0, set it to that number. 0 < Range < 1", min_value=0.0, max_value=1.0, value=0.0)
    max_disc = st.slider("Maximum discrepancy score", help="Maximum discrepancy score to be used for simulation. By default, set to 1 (100% discrepancy between subordinate and supervisor). If you are working with a sector in which you have reason to believe the highest observed discrepancy scores are lower than 1, set it to that number. min_disc < Range < 1", min_value=min_disc, max_value=1.0, value=1.0)
    mean_disc = st.slider("Mean discrepancy score", min_value=min_disc, max_value=max_disc, value=(min_disc + max_disc) / 2)
    std_disc = st.slider("Standard deviation of discrepancy score", min_value=0.0, max_value=(max_disc - min_disc) / 2, value=(max_disc - min_disc) / 4)
    distribution = st.selectbox("Distribution", ["uniform", "normal"], help="Distribution of discrepancy scores to be used for simulation. Currently, only uniform distribution is implemented. We will implement normal and other distributions in future versions.")

    if st.button("Calculate L1 Sample Size"):
        input_data = {
            "min_n_samples": min_n_samples,
            "max_n_samples": max_n_samples,
            "n_subs_per_block": n_subs_per_block,
            "n_blocks_per_district": n_blocks_per_district,
            "n_district": n_district,
            "level_test": level_test,
            "percent_punish": percent_punish,
            "percent_guarantee": percent_guarantee,
            "confidence": confidence,
            "n_simulations": n_simulations,
            "min_disc": min_disc,
            "max_disc": max_disc,
            "mean_disc": mean_disc,
            "std_disc": std_disc,
            "distribution": distribution
        }
        
        # Error handling
        error_status, error_message = check_errors(input_data)
        if error_status == 0:
            st.error(f"Error: {error_message}")
            return

        response = requests.post(L1_SAMPLE_SIZE_ENDPOINT, json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"L1 Sample Size: {result['value']}")
            st.info(result['message'])
        else:
            st.error(f"Error: {response.json()['detail']}")
