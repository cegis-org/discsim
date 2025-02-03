import base64
from io import BytesIO
import streamlit as st
import requests
import os
from PIL import Image
from dotenv import load_dotenv
from src.utils.pre_survey_analysis.error_handling import check_errors

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

THIRD_PARTY_SAMPLING_ENDPOINT = f"{API_BASE_URL}/third-party-sampling"

def third_party_sampling_strategy():
    st.markdown("<h2 style='text-align: center;'>Third-Party Sampling Strategy Predictor", unsafe_allow_html=True)
    with st.form("third_party_sampling_strategy"):
        col1, col2, col3 = st.columns(3)
        with col1:
            total_samples = st.number_input("Total samples", min_value=1, value=100, help="Total data points sampled (100-1000). Range > 0")
        with col2:
            avg_score = st.slider("Avg truth score", 0.0, 1.0, 0.5, help="Expected avg truth score (0.2-0.5). Higher is worse. 0 < Range < 1")
        with col3:
            sd_across = st.number_input("Standard Deviation across blocks", min_value=0.1,  step=0.1, help="Expected std dev of mean truth score across blocks (0.1-0.5). Range > 0")
        
        col4, col5, col6 = st.columns(3)
        with col4:
            sd_within = st.number_input("Standard Deviation within block", min_value=0.1, step= 0.1, help="Expected std dev within a block (0.1-0.5). Range > 0")
        with col5:
            level_test = st.selectbox("Level of test", ["Block", "District", "State"], help="Aggregation level for 3P testing")
        with col6:
            subs_per_block = st.number_input("Subordinates/block", min_value=1, value=10, help="Number of subordinates in a block. Range > 1")
        
        col7, col8, col9 = st.columns(3)
        with col7:
            blocks_per_district = st.number_input("Blocks/district", min_value=1, value=5, help="Number of blocks in a district. Range >= 1")
        with col8:
            districts = st.number_input("Districts", min_value=1, value=1, help="Number of districts. Range >= 1")
        with col9:
            n_simulations = st.number_input("Simulations", min_value=1, value=100, help="Simulation runs (default 100). Higher n gives more accuracy but takes longer. Range > 1")
        
        col10, col11, col12 = st.columns(3)
        with col10:
            min_sub_per_block = st.number_input("Min subordinates/block", min_value=1, value=1, help="Min subordinates measured per block (default 1). 0 < Range < n_sub_per_block")
        with col11:
            percent_blocks_plot = st.slider("% blocks to plot", 0.0, 100.0, 10.0)
        with col12:
            errorbar_type = st.selectbox("Errorbar Type", ["standard deviation", "standard error of the mean", "95% confidence interval"], help="Errorbar Type")

        col13, col14, col15 = st.columns(3)
        with col13:
            n_blocks_reward = st.number_input("Number of Unit Rewarded", min_value=1, value=1, help="The number of units to be rewarded. Depends on level_test and other inputs")
        
        
        if st.form_submit_button("Predict Third-Party Sampling Strategy"):
            input_data = {
                "total_samples": total_samples, "average_truth_score": avg_score,
                "sd_across_blocks": sd_across, "sd_within_block": sd_within,
                "level_test": level_test, "n_subs_per_block": subs_per_block,
                "n_blocks_per_district": blocks_per_district, "n_district": districts,
                "n_simulations": n_simulations, "min_sub_per_block": min_sub_per_block,
                "percent_blocks_plot": percent_blocks_plot, "errorbar_type": errorbar_type,
                "n_blocks_reward": n_blocks_reward
            }
            
            error_status, error_message = check_errors(input_data)
            if error_status == 0:
                st.error(f"Error: {error_message}")
                return

            with st.spinner('Analyzing... Please wait.'):
                response = requests.post(THIRD_PARTY_SAMPLING_ENDPOINT, json=input_data)
            
            if response.status_code == 200:
                result = response.json()
                st.info(result['message'])

                fig1 = base64.b64decode(result['value']['figureImg'])
                image1 = Image.open(BytesIO(fig1))
                st.image(image1, caption="Third-Party Sampling Strategy Plot", use_container_width=True)
                image_bytes = BytesIO()
                image1.save(image_bytes, format="PNG")
                image_bytes.seek(0)
                encoded_image1 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
                download_link1 = f'<a href="data:image/png;base64,{encoded_image1}" class="downloadLink" download="third_party_sampling_plot.png">Click here to download</a>'
                st.markdown(download_link1, unsafe_allow_html=True)

                fig2 = base64.b64decode(result['value']['figure2'])
                image2 = Image.open(BytesIO(fig2))
                st.image(image2, caption="Third-Party Sampling Strategy Plot", use_container_width=True)
                image_bytes2 = BytesIO()
                image2.save(image_bytes2, format="PNG")
                image_bytes2.seek(0)
                encoded_image2 = base64.b64encode(image_bytes2.getvalue()).decode("utf-8")
                download_link2 = f'<a href="data:image/png;base64,{encoded_image2}" class="downloadLink" download="third_party_sampling_plot2.png">Click here to download</a>'
                st.markdown(download_link2, unsafe_allow_html=True)

            else:
                st.error(f"Error: {response.json()['detail']}")
    