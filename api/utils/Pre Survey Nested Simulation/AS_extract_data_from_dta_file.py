import pandas as pd

def extract_data_from_dta(file_path, convert_categoricals=True):
    """
    Extracts data from a .dta file and returns it as a pandas DataFrame.

    Parameters:
        file_path (str): The path to the .dta file.
        convert_categoricals (bool): Whether to convert categorical variables to pandas categories. Set to false if there are columns with repeated values.

    Returns:
        pd.DataFrame: The data extracted from the .dta file.
    """
    try:
        # Read the .dta file using pandas
        data = pd.read_stata(file_path, convert_categoricals= convert_categoricals)
        return data
    except Exception as e:
        print(f"An error occurred while reading the .dta file: {e}")
        return None