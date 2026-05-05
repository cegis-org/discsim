import streamlit as st
import pandas as pd

def display_data_preview(uploaded_file):
    try:
        uploaded_file.seek(0)  # Reset file pointer
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head(), hide_index=True)
    except Exception as e:
        st.error(f"Error reading the CSV file: {str(e)}")
        st.write("Unable to display data preview.")