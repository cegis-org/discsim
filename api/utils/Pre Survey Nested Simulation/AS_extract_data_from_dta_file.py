import pandas as pd

def extract_data_from_dta(file_path):
    """
    Extracts data from a .dta file and returns it as a pandas DataFrame.

    Parameters:
        file_path (str): The path to the .dta file.

    Returns:
        pd.DataFrame: The data extracted from the .dta file.
    """
    try:
        # Read the .dta file using pandas
        data = pd.read_stata(file_path)
        return data
    except Exception as e:
        print(f"An error occurred while reading the .dta file: {e}")
        return None