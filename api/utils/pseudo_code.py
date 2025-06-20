import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

def error_handling(params):
    # This function can be expanded for more specific error logging or custom error responses
    return (1, "Success") # Currently, it always returns success.

def excel_percentrank_inc(series, value):
    """
    Calculates the percentile rank of a value in a pandas Series,
    mimicking Excel's PERCENTRANK.INC function.
    Handles non-numeric values, empty series, and division by zero.
    """
    try:
        if not isinstance(series, pd.Series):
            raise TypeError("Input 'series' must be a pandas Series.")
        
        # Coerce to numeric and drop NaNs to ensure valid ranking
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if numeric_series.empty:
            return 0 # No valid numeric values to rank against

        if pd.isna(value) or value not in numeric_series.values:
            return 0  # Handle missing/irrelevant values (like Excel)
        
        ranked = numeric_series.rank(method="min")  # Match Excel's tie behavior
        count = len(numeric_series)
        
        # Avoid division by zero if there's only one or zero valid data points
        if count <= 1:
            return 0
            
        # Get the rank of the specific value
        # .iloc[0] is used because series == value might return multiple True for duplicate values,
        # and rank() assigns the same rank to ties based on 'min' method.
        rank_of_value = ranked[numeric_series == value].iloc[0]

        return round((rank_of_value - 1) / (count - 1) * 100, 1)
    except Exception as e:
        # Log the error or handle it as appropriate
        print(f"Error in excel_percentrank_inc: {e}")
        return 0 # Return a default or error value

def anganwadi_center_data_anaylsis(file: pd.DataFrame):
    """
    Performs comprehensive data analysis on Anganwadi Center data,
    calculating various metrics and classifications at district, project,
    sector, and AWC levels. Includes robust error handling.

    Args:
        file (pd.DataFrame): The input DataFrame containing Anganwadi data.

    Returns:
        tuple: A tuple containing:
            - int: 1 for success, 0 for failure.
            - str: A success or error message.
            - dict: The analysis results if successful, an empty list if failure.
    """
    try:
        # Working on a copy to avoid modifying the original DataFrame
        df = file.copy() 

        # all required columns for the analysis
        required_columns = [
            'Status_Wasting', 'Sup_Status_Wasting', 
            'Status_UW', 'Sup_Status_UW', 
            'Height', 'Sup_Height', 
            'Weight', 'Sup_Weight', 
            'Muac', 'Sup_Muac', 
            'AWC_ID', 'Sec_ID', 'Proj_Name', 'D_Name','AWC_Name',
            'WeightDate', 'Sup_WeightDate',
            'Status_Stunting', 'Sup_Status_Stunting', 
            'AgeinMonthsAsDate'
        ]

        # Check for missing required columns upfront
        for col in required_columns:
            if col not in df.columns:
                return (0, f"Error: Required column '{col}' is missing from the data. Please ensure all necessary columns are present.", [])
        
        # Convert relevant columns to numeric type to prevent 'unsupported operand type' errors
        # 'errors='coerce'' will turn non-numeric values into NaN
        numeric_cols_to_convert = ['Height', 'Sup_Height', 'Weight', 'Sup_Weight', 'Muac', 'Sup_Muac']
        for col in numeric_cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- Mismatch Classification Conditions ---
        try:

            # Wasting mismatches between AWT and Supervisor
            df['AWT_Normal_Sup_SAM'] = ((df['Sup_Status_Wasting'] == "SAM") & (df['Status_Wasting'] == "Normal")) * 1
            df['AWT_Normal_Sup_MAM'] = ((df['Sup_Status_Wasting'] == "MAM") & (df['Status_Wasting'] == "Normal")) * 1
            df['AWT_MAM_Sup_SAM'] = ((df['Sup_Status_Wasting'] == "SAM") & (df['Status_Wasting'] == "MAM")) * 1

            # Underweight mismatches between AWT and Supervisor
            df['AWT_Normal_Sup_SUW'] = ((df['Sup_Status_UW'] == "SUW") & (df['Status_UW'] == "Normal")) * 1
            df['AWT_Normal_Sup_MUW'] = ((df['Sup_Status_UW'] == "MUW") & (df['Status_UW'] == "Normal")) * 1
            
            # Exact matches in height and weight measurements
            df['AWT_height_eq_Sup_height'] = (df['Height'] == df['Sup_Height']) * 1
            df['AWT_weight_eq_Sup_weight'] = (df['Weight'] == df['Sup_Weight']) * 1
            df['AWT_height_weight_eq_Sup'] = ((df['Height'] == df['Sup_Height']) & (df['Weight'] == df['Sup_Weight'])) * 1
            
            # % Mismatch Discrepancy Condition - If any of the above conditions are true
            df['Discrepancy'] = (
                df['AWT_Normal_Sup_SAM'] | 
                df['AWT_Normal_Sup_MAM'] | 
                df['AWT_MAM_Sup_SAM'] |
                df['AWT_Normal_Sup_SUW'] | 
                df['AWT_Normal_Sup_MUW']
            )
        except Exception as e:
            logger.error(f"Error during mismatch classification calculations: {e}")
            return (0, f"Error during mismatch classification calculations", [])

        # --- Wasting Conditions ---
        try:

            # Identify children classified as wasted by AWT or Supervisor
            df['AWT_Wasting'] = ((df['Status_Wasting'] == "SAM") | (df['Status_Wasting'] == "MAM")) * 1
            df['Supervisor_Wasting'] = ((df['Sup_Status_Wasting'] == "SAM") | (df['Sup_Status_Wasting'] == "MAM")) * 1
            df['AWT_Sup_Same_Wasting'] = (df['Status_Wasting'] == df['Sup_Status_Wasting']) * 1
            df['AWT_Sup_Other_Misclassifications_Wasting'] = (
                (df['AWT_Sup_Same_Wasting'] == 0) & (df['AWT_Normal_Sup_SAM'] == 0) & (df['AWT_Normal_Sup_MAM'] == 0) & (df['AWT_MAM_Sup_SAM'] == 0)) * 1
        except Exception as e:
            logger.error(f"Error during wasting status calculations: {e}")
            return (0, f"Error during wasting status calculations", [])
        
        # --- Underweight Conditions ---
        try:
            
            # Identify children classified as underweight by AWT or Supervisor
            df['AWT_Underweight'] = ((df['Status_UW'] == "SUW") | (df['Status_UW'] == "MUW")) * 1
            df['Supervisor_Underweight'] = ((df['Sup_Status_UW'] == "SUW") | (df['Sup_Status_UW'] == "MUW")) * 1
            df['AWT_Sup_Same_Underweight'] = (df['Status_UW'] == df['Sup_Status_UW']) * 1
            df['AWT_Sup_Other_Misclassifications_Underweight'] = ((df['AWT_Sup_Same_Underweight'] == 0) & (df['AWT_Normal_Sup_SUW'] == 0) & (df['AWT_Normal_Sup_MUW'] == 0)) * 1
        except Exception as e:
            logger.error(f"Error during underweight status calculations: {e}")
            return (0, f"Error during underweight status calculations", [])

        # --- Date Conversions ---
        try:
            # Coerce errors will turn unparseable dates into NaT (Not a Time)
            df['WeightDate'] = pd.to_datetime(df['WeightDate'], format='%d/%m/%Y', errors='coerce')
            df['Sup_WeightDate'] = pd.to_datetime(df['Sup_WeightDate'], format='%d/%m/%Y', errors='coerce')
            df['Gap between AWT Sup Measurements'] = (df['Sup_WeightDate'] - df['WeightDate']).dt.days
        except Exception as e:
            logger.error(f"Error during date conversions and gap calculation: {e}")
            return (0, f"Error during date conversions and gap calculation", [])

        # --- Stunting Conditions ---
        try:
            df['AWT_Stunting'] = ((df['Status_Stunting'] == 'MAM') | (df['Status_Stunting'] == 'SAM')) * 1
            df['Supervisor_Stunting'] = ((df['Sup_Status_Stunting'] == 'MAM') | (df['Sup_Status_Stunting'] == 'SAM')) * 1
            df['AWT_Normal_Sup_Stunt_SAM'] = ((df['Status_Stunting'] == 'Normal') & (df['Sup_Status_Stunting'] == 'SAM')) * 1
            df['AWT_Normal_Sup_Stunt_MAM'] = ((df['Status_Stunting'] == 'Normal') & (df['Sup_Status_Stunting'] == 'MAM')) * 1
            df['AWT_MAM_Sup_Stunt_SAM'] = ((df['Status_Stunting'] == 'MAM') & (df['Sup_Status_Stunting'] == 'SAM')) * 1
            df['AWT_Sup_Same_Stunting'] = (df['Status_Stunting'] == df['Sup_Status_Stunting']) * 1
            df['AWT_Sup_Other_Misclassifications_Stunting'] =(
                (df['AWT_Sup_Same_Stunting'] == 0) & 
                (df['AWT_Normal_Sup_Stunt_MAM'] == 0) & 
                (df['AWT_Normal_Sup_Stunt_SAM'] == 0)
            ) * 1
        except Exception as e:
            logger.error(f"Error during stunting status calculations: {e}")
            return (0, f"Error during stunting status calculations:", [])
        
        # --- Height-Weight Diff ---
        try:
            # Ensure Height and Sup_Height are not NaN and not zero before calculating difference
            df['AWT_Sup_Height_Difference'] = np.round(np.where(
                (df['Height'].notna()) & (df['Sup_Height'].notna()) & 
                (df['Height'] != 0) & (df['Sup_Height'] != 0) & 
                (df['Height'] != df['Sup_Height']),
                abs(df['Sup_Height'] - df['Height']), np.nan), 1
            )
            df['AWT_Sup_Weight_Difference'] = np.where(
                (df['Weight'].notna()) & (df['Sup_Weight'].notna()) & 
                (df['Weight'] != 0) & (df['Sup_Weight'] != 0) & 
                (df['Weight'] != df['Sup_Weight']), 
                abs(df['Weight'] - df['Sup_Weight']), np.nan
            )
        except Exception as e:
            logger.error(f"Error calculating height/weight differences: {e}")
            return (0, f"Error calculating height/weight differences", [])

        # --- Children Classifications ---
        try:
            df['0-3 years old'] = (df['AgeinMonthsAsDate'] < 36) * 1
            df['3-6 years old'] = np.where(df['0-3 years old'] == 0, 1, 0)
        except Exception as e:
            logger.error(f"Error during children age classification: {e}")
            return (0, f"Error during children age classification", [])

        # --- Numeric Column Differences (Height, Weight, Muac) ---
        try:
            numeric_cols = ["Height", "Weight", "Muac"]
            df['Height_Same'] = (df['Height'] == df['Sup_Height']) * 1
            df['Weight_Same'] = (df['Weight'] == df['Sup_Weight']) * 1
            df['Muac_Same'] = (df['Muac'] == df['Sup_Muac']) * 1
            for col in numeric_cols:
                # Ensure columns exist before attempting subtraction
                if col in df.columns and f'Sup_{col}' in df.columns:
                    df[f'{col}_Diff'] = abs(df[col] - df[f'Sup_{col}'])
                else:
                    print(f"Warning: Missing column for {col}_Diff calculation.") # Or handle more strictly
        except Exception as e:
            logger.error(f"Error calculating numeric column differences (Height, Weight, Muac): {e}")
            return (0, f"Error calculating numeric column differences (Height, Weight, Muac)", [])

        # --- Sample Size Check ---
        num_remeasurements = df.shape[0]
        if num_remeasurements == 0:
            return (0, "Error: The input data contains no records for analysis after initial processing.", [])

        # --- Exact Same Height and Weight - AW and Supervisor ---
        try:
            same_values_data = {
                "Metric": ["Exact same height", "Exact same weight"],
                "Value": [
                    (df['Height'] == df['Sup_Height']).sum(),
                    (df['Weight'] == df['Sup_Weight']).sum()
                ]
            }
            same_values_df = pd.DataFrame(same_values_data)
            same_values_df['Percentage (%)'] = round((same_values_df['Value'] / num_remeasurements) * 100, 1)
            same_height_data = df[df['Height'] == df['Sup_Height']]
            same_weight_data = df[df['Weight'] == df['Sup_Weight']]
        except Exception as e:
            logger.error(f"Error calculating exact same height/weight metrics: {e}")
            return (0, f"Error calculating exact same height/weight metrics", [])

        # --- Children Classifications Metrics ---
        try:
            children_category_data = pd.DataFrame({
                "Category": ["0-3 years old", "3-6 years old"],
                "Count": [
                    df['0-3 years old'].sum(),
                    df['3-6 years old'].sum()
                ],
                "Average Height Difference (cms)": [
                    # Check if the filtered series is empty before calling .mean()
                    round(df.loc[df['0-3 years old'] == 1, 'AWT_Sup_Height_Difference'].mean(), 1) 
                    if not df.loc[df['0-3 years old'] == 1, 'AWT_Sup_Height_Difference'].empty else 0,
                    round(df.loc[df['3-6 years old'] == 1, 'AWT_Sup_Height_Difference'].mean(), 1) 
                    if not df.loc[df['3-6 years old'] == 1, 'AWT_Sup_Height_Difference'].empty else 0
                ],
                "Average Weight Difference (kgs)": [
                    round(df.loc[df['0-3 years old'] == 1, 'AWT_Sup_Weight_Difference'].mean(), 1) 
                    if not df.loc[df['0-3 years old'] == 1, 'AWT_Sup_Weight_Difference'].empty else 0,
                    round(df.loc[df['3-6 years old'] == 1, 'AWT_Sup_Weight_Difference'].mean(), 1) 
                    if not df.loc[df['3-6 years old'] == 1, 'AWT_Sup_Weight_Difference'].empty else 0
                ],
                "Average gap in measurement (days)": [
                    round(df.loc[df['0-3 years old'] == 1, 'Gap between AWT Sup Measurements'].mean(), 0) 
                    if not df.loc[df['0-3 years old'] == 1, 'Gap between AWT Sup Measurements'].empty else 0,
                    round(df.loc[df['3-6 years old'] == 1, 'Gap between AWT Sup Measurements'].mean(), 0) 
                    if not df.loc[df['3-6 years old'] == 1, 'Gap between AWT Sup Measurements'].empty else 0
                ]
            })
        except Exception as e:
            logger.error(f"Error calculating children category metrics: {e}")
            return (0, f"Error calculating children category metrics", [])
            
        # --- Wasting Levels ---
        try:
            wasting_metrics_data = {
                "Metric": ["AWT SAM", "Supervisor SAM", "AWT Wasting", "Supervisor Wasting"],
                "Value": [
                    df['Status_Wasting'].eq('SAM').sum(),
                    df['Sup_Status_Wasting'].eq('SAM').sum(),
                    df['AWT_Wasting'].sum(),
                    df['Supervisor_Wasting'].sum(),
                ]
            }
            wasting_metrics_df = pd.DataFrame(wasting_metrics_data)
            wasting_metrics_df['Percentage (%)'] = round((wasting_metrics_df['Value'] / num_remeasurements) * 100, 1)
        except Exception as e:
            logger.error(f"Error calculating wasting levels: {e}")
            return (0, f"Error calculating wasting levels", [])

        # --- Wasting Classifications ---
        try:
            misclassification_wasting_data = {
                "Metric": ["AWT Normal; Supervisor SAM", "AWT Normal; Supervisor MAM", "AWT MAM; Supervisor SAM", "Other Misclassifications", "Same Classification"],
                "Value": [
                    df['AWT_Normal_Sup_SAM'].sum(),
                    df['AWT_Normal_Sup_MAM'].sum(),
                    df['AWT_MAM_Sup_SAM'].sum(),
                    df['AWT_Sup_Other_Misclassifications_Wasting'].sum(),
                    df['AWT_Sup_Same_Wasting'].sum()
                ]
            }
            misclassification_wasting_df = pd.DataFrame(misclassification_wasting_data)
            misclassification_wasting_df['Percentage (%)'] = round((misclassification_wasting_df['Value'] / num_remeasurements) * 100, 1)
            misclassification_wasting_AWT_Normal_Supervisor_SAM = df[(df['Status_Wasting'] == "Normal") & (df['Sup_Status_Wasting'] == "SAM")]
            misclassification_wasting_AWT_Normal_Supervisor_MAM = df[(df['Status_Wasting'] == "Normal") & (df['Sup_Status_Wasting'] == "MAM")]
        except Exception as e:
            logger.error(f"Error calculating wasting misclassifications: {e}")
            return (0, f"Error calculating wasting misclassifications", [])

        # --- UnderWeight Levels ---
        try:
            underweight_metrics_data = {
                "Metric": ["AWT SUW", "Supervisor SUW", "AWT UW", "Supervisor UW"],
                "Value": [
                    df['Status_UW'].eq('SUW').sum(),
                    df['Sup_Status_UW'].eq('SUW').sum(),
                    df['AWT_Underweight'].sum(),
                    df['Supervisor_Underweight'].sum()
                ]
            }
            underweight_metrics_df = pd.DataFrame(underweight_metrics_data)
            underweight_metrics_df['Percentage (%)'] = round((underweight_metrics_df['Value'] / num_remeasurements) * 100, 1)
        except Exception as e:
            logger.error(f"Error calculating underweight levels: {e}")
            return (0, f"Error calculating underweight levels", [])

        # --- Underweight Misclassification ---
        try:
            underweight_classification_data = {
                "Metric": [
                    "AWT Normal; Supervisor SUW",
                    "AWT Normal; Supervisor MUW",
                    "Other Misclassifications", 
                    "Same Classification",
                ],
                "Value": [
                    df['AWT_Normal_Sup_SUW'].sum(),
                    df['AWT_Normal_Sup_MUW'].sum(),
                    df['AWT_Sup_Other_Misclassifications_Underweight'].sum(),
                    df['AWT_Sup_Same_Underweight'].sum(),
                ]
            }
            underweight_classification_df = pd.DataFrame(underweight_classification_data)
            underweight_classification_df['Percentage (%)'] = round((underweight_classification_df['Value'] / num_remeasurements) * 100, 1)
            underweight_classification_AWT_Normal_Supervisor_SUW = df[(df['Status_UW'] == "Normal") & (df['Sup_Status_UW'] == "SUW")]
            underweight_classification_AWT_Normal_Supervisor_MUW = df[(df['Status_UW'] == "Normal") & (df['Sup_Status_UW'] == "MUW")]
        except Exception as e:
            logger.error(f"Error calculating underweight misclassifications: {e}")
            return (0, f"Error calculating underweight misclassifications", [])

        # --- Stunting Levels ---
        try:
            stunting_metrics_data = {
                "Metric": ["AWT SS", "Supervisor SS", "AWT Stunting", "Supervisor Stunting"],
                "Value": [
                    df['Status_Stunting'].eq('SAM').sum(),
                    df['Sup_Status_Stunting'].eq('SAM').sum(),
                    df['AWT_Stunting'].sum(),
                    df['Supervisor_Stunting'].sum()
                ]
            }
            stunting_metrics_df = pd.DataFrame(stunting_metrics_data)
            stunting_metrics_df['Percentage (%)'] = round((stunting_metrics_df['Value'] / num_remeasurements) * 100, 1)
            
            misclassification_stunting_data = {
                "Metric": ["AWT Normal; Supervisor SS", "AWT Normal; Supervisor MS", "AWT MS; Supervisor SS", "Other Misclassifications", "Same Classifications"],
                "Value": [
                    df['AWT_Normal_Sup_Stunt_SAM'].sum(),
                    df['AWT_Normal_Sup_Stunt_MAM'].sum(),
                    df['AWT_MAM_Sup_Stunt_SAM'].sum(),
                    df['AWT_Sup_Other_Misclassifications_Stunting'].sum(),
                    df['AWT_Sup_Same_Stunting'].sum()
                ]
            }
            misclassification_stunting_df = pd.DataFrame(misclassification_stunting_data)
            misclassification_stunting_df['Percentage (%)'] = round((misclassification_stunting_df['Value'] / num_remeasurements) * 100, 1)
        except Exception as e:
            logger.error(f"Error calculating stunting levels/misclassifications: {e}")
            return (0, f"Error calculating stunting levels/misclassifications", [])

        # --- Project Level Analysis ---
        try:
            # Equal Same Height
            project_analysis_eq_height = df.groupby('Proj_Name').agg(
                Total_Remeasurements=('Height', 'count'),
                Exact_Same_Height=('AWT_height_eq_Sup_height', 'sum')
            ).reset_index()
            project_analysis_eq_height['Exact_Same_Height_%'] = round((project_analysis_eq_height['Exact_Same_Height'] / project_analysis_eq_height['Total_Remeasurements']) * 100, 1)

            # Equal Same Weight
            project_analysis_eq_weight = df.groupby('Proj_Name').agg(
                Total_Remeasurements=('Weight', 'count'),
                Exact_Same_Weight=('AWT_weight_eq_Sup_weight', 'sum')
            ).reset_index()
            project_analysis_eq_weight['Exact_Same_Weight_%'] = round((project_analysis_eq_weight['Exact_Same_Weight'] / project_analysis_eq_weight['Total_Remeasurements']) * 100, 1)

            # Wasting Classification
            project_analysis_wasting_classification = df.groupby('Proj_Name').agg(
                Total_Remeasurements=('Proj_Name', 'count'),
                AWT_Normal_Sup_SAM=('AWT_Normal_Sup_SAM', 'sum'),
                AWT_Normal_Sup_MAM=('AWT_Normal_Sup_MAM', 'sum'),
                AWT_MAM_Sup_SAM=('AWT_MAM_Sup_SAM', 'sum'),
                Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Wasting', 'sum'),
                Same_Classifications=('AWT_Sup_Same_Wasting', 'sum'),
            ).reset_index()
            project_analysis_wasting_classification['AWT_Normal_Sup_SAM_%'] = round((project_analysis_wasting_classification['AWT_Normal_Sup_SAM'] / project_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            project_analysis_wasting_classification['AWT_Normal_Sup_MAM_%'] = round((project_analysis_wasting_classification['AWT_Normal_Sup_MAM'] / project_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            project_analysis_wasting_classification['AWT_MAM_Sup_SAM_%'] = round((project_analysis_wasting_classification['AWT_MAM_Sup_SAM'] / project_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            project_analysis_wasting_classification['Other_Misclassifications_%'] = round((project_analysis_wasting_classification['Other_Misclassifications'] / project_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            project_analysis_wasting_classification['Same_Classifications_%'] = round((project_analysis_wasting_classification['Same_Classifications'] / project_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)

            # Wasting Levels
            project_analysis_wasting_levels = df.groupby('Proj_Name').agg(
                Total_Remeasurements=('Proj_Name', 'count'),
                AWT_SAM=('Status_Wasting', lambda x: (x == 'SAM').sum()),
                AWT_Wasting=('AWT_Wasting', 'sum'),
                Supervisor_SAM=('Sup_Status_Wasting', lambda x: (x == 'SAM').sum()),
                Supervisor_Wasting=('Supervisor_Wasting', 'sum')
            ).reset_index()
            project_analysis_wasting_levels['AWT_SAM_%'] = round((project_analysis_wasting_levels['AWT_SAM'] / project_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
            project_analysis_wasting_levels['AWT_Wasting_%'] = round((project_analysis_wasting_levels['AWT_Wasting'] / project_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
            project_analysis_wasting_levels['Supervisor_SAM_%'] = round((project_analysis_wasting_levels['Supervisor_SAM'] / project_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
            project_analysis_wasting_levels['Supervisor_Wasting_%'] = round((project_analysis_wasting_levels['Supervisor_Wasting'] / project_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
            project_analysis_wasting_levels['Sup-AWT_Difference_%'] = round(((project_analysis_wasting_levels['Supervisor_Wasting'] - project_analysis_wasting_levels['AWT_Wasting']) / project_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)

            # Underweight Classification
            project_analysis_underweight_classification = df.groupby('Proj_Name').agg(
                Total_Remeasurements=('Proj_Name', 'count'),
                AWT_Normal_Sup_SUW=('AWT_Normal_Sup_SUW', 'sum'),
                AWT_Normal_Sup_MUW=('AWT_Normal_Sup_MUW', 'sum'),
                Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Underweight', 'sum'),
                Same_Classifications=('AWT_Sup_Same_Underweight', 'sum'),
            ).reset_index()
            project_analysis_underweight_classification['AWT_Normal_Sup_SUW_%'] = round((project_analysis_underweight_classification['AWT_Normal_Sup_SUW'] / project_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
            project_analysis_underweight_classification['AWT_Normal_Sup_MUW_%'] = round((project_analysis_underweight_classification['AWT_Normal_Sup_MUW'] / project_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
            project_analysis_underweight_classification['Other_Misclassifications_%'] = round((project_analysis_underweight_classification['Other_Misclassifications'] / project_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
            project_analysis_underweight_classification['Same_Classifications_%'] = round((project_analysis_underweight_classification['Same_Classifications'] / project_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)

            # Underweight Levels
            project_analysis_uw_levels = df.groupby('Proj_Name').agg(
                Total_Remeasurements=('Proj_Name', 'count'),
                AWT_SUW=('Status_UW', lambda x: (x == 'SUW').sum()),
                Sup_SUW=('Sup_Status_UW', lambda x: (x == 'SUW').sum()),
                AWT_Underweight=('AWT_Underweight', 'sum'),
                Sup_Underweight=('Supervisor_Underweight', 'sum')
            ).reset_index()
            project_analysis_uw_levels['AWT_SUW_%'] = round((project_analysis_uw_levels['AWT_SUW'] / project_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
            project_analysis_uw_levels['Sup_SUW_%'] = round((project_analysis_uw_levels['Sup_SUW'] / project_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
            project_analysis_uw_levels['AWT_Underweight_%'] = round((project_analysis_uw_levels['AWT_Underweight'] / project_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
            project_analysis_uw_levels['Sup_Underweight_%'] = round((project_analysis_uw_levels['Sup_Underweight'] / project_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
            project_analysis_uw_levels['Sup-AWT_Difference_%'] = round(((project_analysis_uw_levels['Sup_Underweight'] - project_analysis_uw_levels['AWT_Underweight']) / project_analysis_uw_levels['Total_Remeasurements']) * 100, 0)

            project_stunting_level =  df.groupby('Proj_Name').agg(
                Total_Remeasurements=('Proj_Name', 'count'),
                AWT_SS=('Status_Stunting', lambda x: (x == 'SAM').sum()),
                Sup_SS=('Sup_Status_Stunting', lambda x: (x == 'SAM').sum()),
                AWT_Stunting=('AWT_Stunting', 'sum'),
                Sup_Stunting=('Supervisor_Stunting', 'sum')
            ).reset_index()
            project_stunting_level['AWT_SS_%'] = round((project_stunting_level['AWT_SS'] / project_stunting_level['Total_Remeasurements']) * 100, 0)
            project_stunting_level['Sup_SS_%'] = round((project_stunting_level['Sup_SS'] / project_stunting_level['Total_Remeasurements']) * 100, 0)
            project_stunting_level['AWT_Stunting_%'] = round((project_stunting_level['AWT_Stunting'] / project_stunting_level['Total_Remeasurements']) * 100, 0)
            project_stunting_level['Sup_Stunting_%'] = round((project_stunting_level['Sup_Stunting'] / project_stunting_level['Total_Remeasurements']) * 100, 0)

            project_stunting_classification = df.groupby('Proj_Name').agg(
                Total_Remeasurements=('Proj_Name', 'count'),
                AWT_Normal_Sup_SS=('AWT_Normal_Sup_Stunt_SAM', 'sum'),
                AWT_Normal_Sup_MS=('AWT_Normal_Sup_Stunt_MAM', 'sum'),
                AWT_MS_Sup_SS=('AWT_MAM_Sup_Stunt_SAM', 'sum'),
                Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Stunting', 'sum'),
                Same_Classifications=('AWT_Sup_Same_Stunting', 'sum'),
            ).reset_index()
            project_stunting_classification['AWT_Normal_Sup_SS_%'] = round((project_stunting_classification['AWT_Normal_Sup_SS'] / project_stunting_classification['Total_Remeasurements']) * 100, 0)
            project_stunting_classification['AWT_Normal_Sup_MS_%'] = round((project_stunting_classification['AWT_Normal_Sup_MS'] / project_stunting_classification['Total_Remeasurements']) * 100, 0)
            project_stunting_classification['AWT_MS_Sup_SS_%'] = round((project_stunting_classification['AWT_MS_Sup_SS'] / project_stunting_classification['Total_Remeasurements']) * 100, 0)
            project_stunting_classification['Other_Misclassifications_%'] = round((project_stunting_classification['Other_Misclassifications'] / project_stunting_classification['Total_Remeasurements']) * 100, 0)
            project_stunting_classification['Same_Classifications_%'] = round((project_stunting_classification['Same_Classifications'] / project_stunting_classification['Total_Remeasurements']) * 100, 0)

            # Project Level Discrepancy
            project_level_disc = df.groupby('Proj_Name').agg(
                Total_Remeasurements=('Proj_Name', 'count'),
                AWT_Normal_Sup_SAM=('AWT_Normal_Sup_SAM', 'sum'),
                AWT_Normal_Sup_MAM=('AWT_Normal_Sup_MAM', 'sum'),
                AWT_MAM_Sup_SAM=('AWT_MAM_Sup_SAM', 'sum'),
                AWT_Normal_Sup_SUW=('AWT_Normal_Sup_SUW', 'sum'),
                AWT_Normal_Sup_MUW=('AWT_Normal_Sup_MUW', 'sum'),
                Discrepancy_remeasurements=('Discrepancy', 'sum'),
            ).reset_index()
            project_level_disc['Discrepancy Rate (%)'] = np.where(project_level_disc['Total_Remeasurements'] > 15,round((project_level_disc['Discrepancy_remeasurements'] / project_level_disc['Total_Remeasurements']) * 100,1),0)
            project_level_disc['Non-Discrepancy Rate (%)'] = np.where(project_level_disc['Total_Remeasurements'] > 15,round(100 - project_level_disc['Discrepancy Rate (%)'],1),0)
            
            # Ensure valid_disc_rates is not empty before applying excel_percentrank_inc
            valid_disc_rates = project_level_disc[project_level_disc['Discrepancy Rate (%)'] > 0]['Non-Discrepancy Rate (%)']
            if not valid_disc_rates.empty:
                project_level_disc["Percentile_Rank (%)"] = project_level_disc["Non-Discrepancy Rate (%)"].apply(
                    lambda x: excel_percentrank_inc(valid_disc_rates, x) if x > 0 else 0
                )
            else:
                project_level_disc["Percentile_Rank (%)"] = 0 # Default if no valid rates

            project_level_disc['Zone'] = np.where(
            project_level_disc['Discrepancy Rate (%)'] > 0,
            np.select(
                [
                    project_level_disc['Percentile_Rank (%)'] >= 75, #green threshold
                    project_level_disc['Percentile_Rank (%)'] <= 25 #red threshold
                ], ['Green', 'Red'],default='Yellow'),'')
        except Exception as e:
            logger.error(f"Error during Project Level Analysis: {e}")
            return (0, f"Error during Project Level Analysis", [])

        # --- Sector Level Analysis ---
        try:
            # Equal Same Height
            sector_analysis_eq_height = df.groupby('Sec_Name').agg(
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Height', 'count'),
                Exact_Same_Height=('AWT_height_eq_Sup_height', 'sum')
            ).reset_index()
            sector_analysis_eq_height['Exact_Same_Height_%'] = np.where(sector_analysis_eq_height['Total_Remeasurements'] > 15,round((sector_analysis_eq_height['Exact_Same_Height'] / sector_analysis_eq_height['Total_Remeasurements']) * 100, 1),0)

            # Equal Same Weight
            sector_analysis_eq_weight = df.groupby('Sec_Name').agg(
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Weight', 'count'),
                Exact_Same_Weight=('AWT_weight_eq_Sup_weight', 'sum')
            ).reset_index()
            sector_analysis_eq_weight['Exact_Same_Weight_%'] = np.where(sector_analysis_eq_weight['Total_Remeasurements'] > 15,round((sector_analysis_eq_weight['Exact_Same_Weight'] / sector_analysis_eq_weight['Total_Remeasurements']) * 100, 1),0) 
            #15 is case-threshold for user input (can be changed at each level)

            # Wasting Levels
            sector_analysis_wasting_levels = df.groupby('Sec_Name').agg(
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Sec_Name', 'count'),
                AWT_SAM=('Status_Wasting', lambda x: (x == 'SAM').sum()),
                AWT_Wasting=('AWT_Wasting', 'sum'),
                Supervisor_SAM=('Sup_Status_Wasting', lambda x: (x == 'SAM').sum()),
                Supervisor_Wasting=('Supervisor_Wasting', 'sum')
            ).reset_index()
            sector_analysis_wasting_levels['AWT_SAM_%'] = round((sector_analysis_wasting_levels['AWT_SAM'] / sector_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
            sector_analysis_wasting_levels['AWT_Wasting_%'] = round((sector_analysis_wasting_levels['AWT_Wasting'] / sector_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
            sector_analysis_wasting_levels['Supervisor_SAM_%'] = round((sector_analysis_wasting_levels['Supervisor_SAM'] / sector_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
            sector_analysis_wasting_levels['Supervisor_Wasting_%'] = round((sector_analysis_wasting_levels['Supervisor_Wasting'] / sector_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
            sector_analysis_wasting_levels['Sup-AWT_Difference_%'] = round(((sector_analysis_wasting_levels['Supervisor_Wasting'] - sector_analysis_wasting_levels['AWT_Wasting']) / sector_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)

            # Wasting Classification
            sector_analysis_wasting_classification = df.groupby('Sec_Name').agg(
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Sec_Name', 'count'),
                AWT_Normal_Sup_SAM=('AWT_Normal_Sup_SAM', 'sum'),
                AWT_Normal_Sup_MAM=('AWT_Normal_Sup_MAM', 'sum'),
                AWT_MAM_Sup_SAM=('AWT_MAM_Sup_SAM', 'sum'),
                Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Wasting', 'sum'),
                Same_Classifications=('AWT_Sup_Same_Wasting', 'sum'),
            ).reset_index()
            sector_analysis_wasting_classification['AWT_Normal_Sup_SAM_%'] = round((sector_analysis_wasting_classification['AWT_Normal_Sup_SAM'] / sector_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            sector_analysis_wasting_classification['AWT_Normal_Sup_MAM_%'] = round((sector_analysis_wasting_classification['AWT_Normal_Sup_MAM'] / sector_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            sector_analysis_wasting_classification['AWT_MAM_Sup_SAM_%'] = round((sector_analysis_wasting_classification['AWT_MAM_Sup_SAM'] / sector_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            sector_analysis_wasting_classification['Other_Misclassifications_%'] = round((sector_analysis_wasting_classification['Other_Misclassifications'] / sector_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            sector_analysis_wasting_classification['Same_Classifications_%'] = round((sector_analysis_wasting_classification['Same_Classifications'] / sector_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)

            # Underweight Levels
            sector_analysis_uw_levels = df.groupby('Sec_Name').agg(
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Sec_Name', 'count'),
                AWT_SUW=('Status_UW', lambda x: (x == 'SUW').sum()),
                Sup_SUW=('Sup_Status_UW', lambda x: (x == 'SUW').sum()),
                AWT_Underweight=('AWT_Underweight', 'sum'),
                Sup_Underweight=('Supervisor_Underweight', 'sum')
            ).reset_index()
            sector_analysis_uw_levels['AWT_SUW_%'] = round((sector_analysis_uw_levels['AWT_SUW'] / sector_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
            sector_analysis_uw_levels['Sup_SUW_%'] = round((sector_analysis_uw_levels['Sup_SUW'] / sector_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
            sector_analysis_uw_levels['AWT_Underweight_%'] = round((sector_analysis_uw_levels['AWT_Underweight'] / sector_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
            sector_analysis_uw_levels['Sup_Underweight_%'] = round((sector_analysis_uw_levels['Sup_Underweight'] / sector_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
            sector_analysis_uw_levels['Sup-AWT_Difference_%'] = round(((sector_analysis_uw_levels['Sup_Underweight'] - sector_analysis_uw_levels['AWT_Underweight']) / sector_analysis_uw_levels['Total_Remeasurements']) * 100, 0)

            # Underweight Classification
            sector_analysis_underweight_classification = df.groupby('Sec_Name').agg(
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Sec_Name', 'count'),
                AWT_Normal_Sup_SUW=('AWT_Normal_Sup_SUW', 'sum'),
                AWT_Normal_Sup_MUW=('AWT_Normal_Sup_MUW', 'sum'),
                Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Underweight', 'sum'),
                Same_Classifications=('AWT_Sup_Same_Underweight', 'sum'),
            ).reset_index()
            sector_analysis_underweight_classification['AWT_Normal_Sup_SUW_%'] = round((sector_analysis_underweight_classification['AWT_Normal_Sup_SUW'] / sector_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
            sector_analysis_underweight_classification['AWT_Normal_Sup_MUW_%'] = round((sector_analysis_underweight_classification['AWT_Normal_Sup_MUW'] / sector_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
            sector_analysis_underweight_classification['Other_Misclassifications_%'] = round((sector_analysis_underweight_classification['Other_Misclassifications'] / sector_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
            sector_analysis_underweight_classification['Same_Classifications_%'] = round((sector_analysis_underweight_classification['Same_Classifications'] / sector_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
    
            # Stunting Levels
            sector_stunting_level =  df.groupby('Sec_Name').agg(
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Sec_Name', 'count'),
                AWT_SS=('Status_Stunting', lambda x: (x == 'SAM').sum()),
                Sup_SS=('Sup_Status_Stunting', lambda x: (x == 'SAM').sum()),
                AWT_Stunting=('AWT_Stunting', 'sum'),
                Sup_Stunting=('Supervisor_Stunting', 'sum')
            ).reset_index()
            sector_stunting_level['AWT_SS_%'] = round((sector_stunting_level['AWT_SS'] / sector_stunting_level['Total_Remeasurements']) * 100, 0)
            sector_stunting_level['Sup_SS_%'] = round((sector_stunting_level['Sup_SS'] / sector_stunting_level['Total_Remeasurements']) * 100, 0)
            sector_stunting_level['AWT_Stunting_%'] = round((sector_stunting_level['AWT_Stunting'] / sector_stunting_level['Total_Remeasurements']) * 100, 0)
            sector_stunting_level['Sup_Stunting_%'] = round((sector_stunting_level['Sup_Stunting'] / sector_stunting_level['Total_Remeasurements']) * 100, 0)

            sector_stunting_classification = df.groupby('Sec_Name').agg(
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Sec_Name', 'count'),
                AWT_Normal_Sup_SS=('AWT_Normal_Sup_Stunt_SAM', 'sum'),
                AWT_Normal_Sup_MS=('AWT_Normal_Sup_Stunt_MAM', 'sum'),
                AWT_MS_Sup_SS=('AWT_MAM_Sup_Stunt_SAM', 'sum'),
                Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Stunting', 'sum'),
                Same_Classifications=('AWT_Sup_Same_Stunting', 'sum'),
            ).reset_index()
            sector_stunting_classification['AWT_Normal_Sup_SS_%'] = round((sector_stunting_classification['AWT_Normal_Sup_SS'] / sector_stunting_classification['Total_Remeasurements']) * 100, 0)
            sector_stunting_classification['AWT_Normal_Sup_MS_%'] = round((sector_stunting_classification['AWT_Normal_Sup_MS'] / sector_stunting_classification['Total_Remeasurements']) * 100, 0)
            sector_stunting_classification['AWT_MS_Sup_SS_%'] = round((sector_stunting_classification['AWT_MS_Sup_SS'] / sector_stunting_classification['Total_Remeasurements']) * 100, 0)
            sector_stunting_classification['Other_Misclassifications_%'] = round((sector_stunting_classification['Other_Misclassifications'] / sector_stunting_classification['Total_Remeasurements']) * 100, 0)
            sector_stunting_classification['Same_Classifications_%'] = round((sector_stunting_classification['Same_Classifications'] / sector_stunting_classification['Total_Remeasurements']) * 100, 0)

            #Sector Level Discrepancy
            sector_level_disc = df.groupby('Sec_Name').agg(
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Sec_Name', 'count'),
                AWT_Normal_Sup_SAM=('AWT_Normal_Sup_SAM', 'sum'),
                AWT_Normal_Sup_MAM=('AWT_Normal_Sup_MAM', 'sum'),
                AWT_MAM_Sup_SAM=('AWT_MAM_Sup_SAM', 'sum'),
                AWT_Normal_Sup_SUW=('AWT_Normal_Sup_SUW', 'sum'),
                AWT_Normal_Sup_MUW=('AWT_Normal_Sup_MUW', 'sum'),
                Discrepancy_remeasurements=('Discrepancy', 'sum'),
            ).reset_index()
            sector_level_disc['Discrepancy Rate (%)'] = np.where(sector_level_disc['Total_Remeasurements'] > 15,round((sector_level_disc['Discrepancy_remeasurements'] / sector_level_disc['Total_Remeasurements']) * 100,1),0)
            sector_level_disc['Non-Discrepancy Rate (%)'] = np.where(sector_level_disc['Total_Remeasurements'] > 15,round(100 - sector_level_disc['Discrepancy Rate (%)'],1),0)
            
            valid_disc_rates = sector_level_disc[sector_level_disc['Discrepancy Rate (%)'] > 0]['Non-Discrepancy Rate (%)']
            if not valid_disc_rates.empty:
                sector_level_disc["Percentile_Rank (%)"] = sector_level_disc["Non-Discrepancy Rate (%)"].apply(
                    lambda x: excel_percentrank_inc(valid_disc_rates, x) if x > 0 else 0
                )
            else:
                sector_level_disc["Percentile_Rank (%)"] = 0

            sector_level_disc['Zone'] = np.where(
            sector_level_disc['Discrepancy Rate (%)'] > 0,
            np.select(
                [
                    sector_level_disc['Percentile_Rank (%)'] >= 75, #green threshold
                    sector_level_disc['Percentile_Rank (%)'] <= 25 #red threshold
                ], ['Green', 'Red'],default='Yellow'),'')
        except Exception as e:
            logger.error(f"Error during Sector Level Analysis: {e}")
            return (0, f"Error during Sector Level Analysis", [])
            
        # --- AWC Level Analysis ---
        try:
            # Equal Same Height
            awc_analysis_eq_height = df.groupby('AWC_Name').agg(
                Sector_Name=('Sec_Name', 'first'),
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Height', 'count'),
                Exact_Same_Height=('AWT_height_eq_Sup_height', 'sum')
            ).reset_index()
            awc_analysis_eq_height['Exact_Same_Height_%'] = round((awc_analysis_eq_height['Exact_Same_Height'] / awc_analysis_eq_height['Total_Remeasurements']) * 100, 1)

            # Equal Same Weight
            awc_analysis_eq_weight = df.groupby('AWC_Name').agg(
                Sector_Name=('Sec_Name', 'first'),
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Weight', 'count'),
                Exact_Same_Weight=('AWT_weight_eq_Sup_weight', 'sum')
            ).reset_index()
            awc_analysis_eq_weight['Exact_Same_Weight_%'] = round((awc_analysis_eq_weight['Exact_Same_Weight'] / awc_analysis_eq_weight['Total_Remeasurements']) * 100, 1)

            awc_analysis_eq_height_weight = df.groupby('AWC_Name').agg(
                Sector_Name=('Sec_Name', 'first'),
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('Weight', 'count'),
                Same_Height_Weight=('AWT_height_weight_eq_Sup','sum')
            ).reset_index()
            awc_analysis_eq_height_weight['Same_Height_Weight_%'] = round((awc_analysis_eq_height_weight['Same_Height_Weight'] / awc_analysis_eq_height_weight['Total_Remeasurements']) * 100, 1)

            #Wasting Levels
            awc_analysis_wasting_levels = df.groupby('AWC_Name').agg(
                Sector_Name=('Sec_Name', 'first'),
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('AWC_Name', 'count'),
                AWT_Wasting=('AWT_Wasting', 'sum'),
                Supervisor_Wasting=('Supervisor_Wasting', 'sum')
            ).reset_index()
            awc_analysis_wasting_levels['AWT_Wasting_%'] = round((awc_analysis_wasting_levels['AWT_Wasting'] / awc_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
            awc_analysis_wasting_levels['Supervisor_Wasting_%'] = round((awc_analysis_wasting_levels['Supervisor_Wasting'] / awc_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
            awc_analysis_wasting_levels['Sup-AWT_Difference_%'] = round(((awc_analysis_wasting_levels['Supervisor_Wasting'] - awc_analysis_wasting_levels['AWT_Wasting']) / awc_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)

            # Wasting Classification
            # awc_analysis_wasting_classification = df.groupby('AWC_Name').agg(
            #     Sector_Name=('Sec_Name', 'first'),
            #     Project_Name=('Proj_Name', 'first'),
            #     Total_Remeasurements=('AWC_Name', 'count'),
            #     AWT_Normal_Sup_SAM=('AWT_Normal_Sup_SAM', 'sum'),
            #     AWT_Normal_Sup_MAM=('AWT_Normal_Sup_MAM', 'sum'),
            #     AWT_MAM_Sup_SAM=('AWT_MAM_Sup_SAM', 'sum'),
            #     Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Wasting', 'sum'),
            #     Same_Classifications=('AWT_Sup_Same_Wasting', 'sum'),
            # ).reset_index()
            # awc_analysis_wasting_classification['AWT_Normal_Sup_SAM_%'] = round((awc_analysis_wasting_classification['AWT_Normal_Sup_SAM'] / awc_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            # awc_analysis_wasting_classification['AWT_Normal_Sup_MAM_%'] = round((awc_analysis_wasting_classification['AWT_Normal_Sup_MAM'] / awc_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            # awc_analysis_wasting_classification['AWT_MAM_Sup_SAM_%'] = round((awc_analysis_wasting_classification['AWT_MAM_Sup_SAM'] / awc_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            # awc_analysis_wasting_classification['Other_Misclassifications_%'] = round((awc_analysis_wasting_classification['Other_Misclassifications'] / awc_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
            # awc_analysis_wasting_classification['Same_Classifications_%'] = round((awc_analysis_wasting_classification['Same_Classifications'] / awc_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)

            # Underweight Levels
            awc_analysis_uw_levels = df.groupby('AWC_Name').agg(
                Sector_Name=('Sec_Name', 'first'),
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('AWC_Name', 'count'),
                AWT_Underweight=('AWT_Underweight', 'sum'),
                Sup_Underweight=('Supervisor_Underweight', 'sum')
            ).reset_index()
            awc_analysis_uw_levels['AWT_Underweight_%'] = round((awc_analysis_uw_levels['AWT_Underweight'] / awc_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
            awc_analysis_uw_levels['Sup_Underweight_%'] = round((awc_analysis_uw_levels['Sup_Underweight'] / awc_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
            awc_analysis_uw_levels['Sup-AWT_Difference_%'] = round(((awc_analysis_uw_levels['Sup_Underweight'] - awc_analysis_uw_levels['AWT_Underweight']) / awc_analysis_uw_levels['Total_Remeasurements']) * 100, 0)

            # Underweight Classification
            # awc_analysis_underweight_classification = df.groupby('AWC_Name').agg(
            #     Total_Remeasurements=('AWC_Name', 'count'),
            #     Sector_Name=('Sec_Name', 'first'),
            #     Project_Name=('Proj_Name', 'first'),
            #     AWT_Normal_Sup_SUW=('AWT_Normal_Sup_SUW', 'sum'),
            #     AWT_Normal_Sup_MUW=('AWT_Normal_Sup_MUW', 'sum'),
            #     Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Underweight', 'sum'),
            #     Same_Classifications=('AWT_Sup_Same_Underweight', 'sum'),
            # ).reset_index()
            # awc_analysis_underweight_classification['AWT_Normal_Sup_SUW_%'] = round((awc_analysis_underweight_classification['AWT_Normal_Sup_SUW'] / awc_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
            # awc_analysis_underweight_classification['AWT_Normal_Sup_MUW_%'] = round((awc_analysis_underweight_classification['AWT_Normal_Sup_MUW'] / awc_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
            # awc_analysis_underweight_classification['Other_Misclassifications_%'] = round((awc_analysis_underweight_classification['Other_Misclassifications'] / awc_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
            # awc_analysis_underweight_classification['Same_Classifications_%'] = round((awc_analysis_underweight_classification['Same_Classifications'] / awc_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)

            awc_stunting_level =  df.groupby('AWC_Name').agg(
                Sector_Name=('Sec_Name', 'first'),
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('AWC_Name', 'count'),
                AWT_SS=('Status_Stunting', lambda x: (x == 'SAM').sum()),
                Sup_SS=('Sup_Status_Stunting', lambda x: (x == 'SAM').sum()),
                AWT_Stunting=('AWT_Stunting', 'sum'),
                Sup_Stunting=('Supervisor_Stunting', 'sum')
            ).reset_index()
            awc_stunting_level['AWT_SS_%'] = round((awc_stunting_level['AWT_SS'] / awc_stunting_level['Total_Remeasurements']) * 100, 0)
            awc_stunting_level['Sup_SS_%'] = round((awc_stunting_level['Sup_SS'] / awc_stunting_level['Total_Remeasurements']) * 100, 0)
            awc_stunting_level['AWT_Stunting_%'] = round((awc_stunting_level['AWT_Stunting'] / awc_stunting_level['Total_Remeasurements']) * 100, 0)
            awc_stunting_level['Sup_Stunting_%'] = round((awc_stunting_level['Sup_Stunting'] / awc_stunting_level['Total_Remeasurements']) * 100, 0)

            awc_stunting_classification = df.groupby('AWC_Name').agg(
                Sector_Name=('Sec_Name', 'first'),
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('AWC_Name', 'count'),
                AWT_Normal_Sup_SS=('AWT_Normal_Sup_Stunt_SAM', 'sum'),
                AWT_Normal_Sup_MS=('AWT_Normal_Sup_Stunt_MAM', 'sum'),
                AWT_MS_Sup_SS=('AWT_MAM_Sup_Stunt_SAM', 'sum'),
                Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Stunting', 'sum'),
                Same_Classifications=('AWT_Sup_Same_Stunting', 'sum'),
            ).reset_index()
            awc_stunting_classification['AWT_Normal_Sup_SS_%'] = round((awc_stunting_classification['AWT_Normal_Sup_SS'] / awc_stunting_classification['Total_Remeasurements']) * 100, 0)
            awc_stunting_classification['AWT_Normal_Sup_MS_%'] = round((awc_stunting_classification['AWT_Normal_Sup_MS'] / awc_stunting_classification['Total_Remeasurements']) * 100, 0)
            awc_stunting_classification['AWT_MS_Sup_SS_%'] = round((awc_stunting_classification['AWT_MS_Sup_SS'] / awc_stunting_classification['Total_Remeasurements']) * 100, 0)
            awc_stunting_classification['Other_Misclassifications_%'] = round((awc_stunting_classification['Other_Misclassifications'] / awc_stunting_classification['Total_Remeasurements']) * 100, 0)
            awc_stunting_classification['Same_Classifications_%'] = round((awc_stunting_classification['Same_Classifications'] / awc_stunting_classification['Total_Remeasurements']) * 100, 0)

            #AWC Level Discrepancy
            awc_level_disc = df.groupby('AWC_Name').agg(
                Sector_Name=('Sec_Name', 'first'),
                Project_Name=('Proj_Name', 'first'),
                Total_Remeasurements=('AWC_Name', 'count'),
                AWT_Normal_Sup_SAM=('AWT_Normal_Sup_SAM', 'sum'),
                AWT_Normal_Sup_MAM=('AWT_Normal_Sup_MAM', 'sum'),
                AWT_MAM_Sup_SAM=('AWT_MAM_Sup_SAM', 'sum'),
                AWT_Normal_Sup_SUW=('AWT_Normal_Sup_SUW', 'sum'),
                AWT_Normal_Sup_MUW=('AWT_Normal_Sup_MUW', 'sum'),
                Discrepancy_remeasurements=('Discrepancy', 'sum'),
            ).reset_index()
            awc_level_disc['Discrepancy Rate (%)'] = np.where(awc_level_disc['Total_Remeasurements'] > 5,round((awc_level_disc['Discrepancy_remeasurements'] / awc_level_disc['Total_Remeasurements']) * 100,1),0)
            awc_level_disc['Non-Discrepancy Rate (%)'] = np.where(awc_level_disc['Total_Remeasurements'] > 5,round(100 - awc_level_disc['Discrepancy Rate (%)'],1),0)
            
            valid_disc_rates = awc_level_disc[awc_level_disc['Discrepancy Rate (%)'] > 0]['Non-Discrepancy Rate (%)']
            if not valid_disc_rates.empty:
                awc_level_disc["Percentile_Rank (%)"] = awc_level_disc["Non-Discrepancy Rate (%)"].apply(
                    lambda x: excel_percentrank_inc(valid_disc_rates, x) if x > 0 else 0
                )
            else:
                awc_level_disc["Percentile_Rank (%)"] = 0

            awc_level_disc['Zone'] = np.select(
                [
                    awc_level_disc['Percentile_Rank (%)'] >= 75, #green threshold
                    awc_level_disc['Percentile_Rank (%)'] <= 25 #red threshold
                ], ['Green', 'Red'],default='Yellow')
        except Exception as e:
            logger.error(f"Error during AWC Level Analysis: {e}")
            return (0, f"Error during AWC Level Analysis", [])

        # --- Prepare Response Data ---
        try:
            response_data = {
                "summary":{
                    "totalSampleSize":num_remeasurements,
                    "AWC": df['AWC_ID'].nunique(),
                    "sectors":df['Sec_ID'].nunique(),
                    "projects":df['Proj_Name'].nunique(),
                    "districts":df['D_Name'].nunique()
                },
                "districtLevelInsights":{
                    "sameHeightWeight":same_values_df.to_dict(orient="records"),
                    "sameHeightRecords":json.dumps(same_height_data.to_dict(orient="records"), default=str),
                    "sameWeightRecords":json.dumps(same_weight_data.to_dict(orient="records"), default=str),
                    "childrenCategory":json.dumps(children_category_data.to_dict(orient="records"), default=str),
                    "wastingLevels":wasting_metrics_df.to_dict(orient="records"),
                    "wastingClassification":misclassification_wasting_df.to_dict(orient="records"),
                    "misclassification_wasting_AWT_Normal_Supervisor_SAM":json.dumps(misclassification_wasting_AWT_Normal_Supervisor_SAM.to_dict(orient="records"), default=str),
                    "misclassification_wasting_AWT_Normal_Supervisor_MAM":json.dumps(misclassification_wasting_AWT_Normal_Supervisor_MAM.to_dict(orient="records"), default=str),
                    "underweightLevels":underweight_metrics_df.to_dict(orient="records"),
                    "underweightClassification":underweight_classification_df.to_dict(orient="records"),
                    "underweight_classification_AWT_Normal_Supervisor_SUW":json.dumps(underweight_classification_AWT_Normal_Supervisor_SUW.to_dict(orient="records"), default=str),
                    "underweight_classification_AWT_Normal_Supervisor_MUW":json.dumps(underweight_classification_AWT_Normal_Supervisor_MUW.to_dict(orient="records"), default=str),
                    "stuntingLevels":stunting_metrics_df.to_dict(orient="records"),
                    "stuntingClassification":misclassification_stunting_df.to_dict(orient="records")
                },
                "projectLevelInsights":{
                    "sameHeight":project_analysis_eq_height.to_dict(orient="records"),
                    "sameWeight":project_analysis_eq_weight.to_dict(orient="records"),
                    "wastingLevels":project_analysis_wasting_levels.to_dict(orient="records"),
                    "wastingClassification":project_analysis_wasting_classification.to_dict(orient="records"),
                    "underweightLevels":project_analysis_uw_levels.to_dict(orient="records"),
                    "underweightClassification":project_analysis_underweight_classification.to_dict(orient="records"),
                    "stuntingLevels":project_stunting_level.to_dict(orient="records"),
                    "stuntingClassification":project_stunting_classification.to_dict(orient="records"),
                    "discrepancy":project_level_disc.to_dict(orient="records"),
                },
                "sectorLevelInsights":{
                    "sameHeight":sector_analysis_eq_height.to_dict(orient="records"),
                    "sameWeight":sector_analysis_eq_weight.to_dict(orient="records"),
                    "wastingLevels":sector_analysis_wasting_levels.to_dict(orient="records"),
                    "wastingClassification":sector_analysis_wasting_classification.to_dict(orient="records"),
                    "underweightLevels":sector_analysis_uw_levels.to_dict(orient="records"),
                    "underweightClassification":sector_analysis_underweight_classification.to_dict(orient="records"),
                    "stuntingLevels":sector_stunting_level.to_dict(orient="records"),
                    "stuntingClassification":sector_stunting_classification.to_dict(orient="records"),
                    "discrepancy":sector_level_disc.to_dict(orient="records"),
                },
                "awcLevelInsights":{
                    "sameHeight": json.dumps(awc_analysis_eq_height.to_dict(orient="records"), default=str),
                    "sameWeight": json.dumps(awc_analysis_eq_weight.to_dict(orient="records"), default=str),
                    "sameHeightWeight": json.dumps(awc_analysis_eq_height_weight.to_dict(orient="records"),default=str),
                    "wastingLevels":awc_analysis_wasting_levels.to_dict(orient="records"),
                    # "wastingClassification":awc_analysis_wasting_classification.to_dict(orient="records"),
                    "underweightLevels":awc_analysis_uw_levels.to_dict(orient="records"),
                    # "underweightClassification":awc_analysis_underweight_classification.to_dict(orient="records"),
                    "stuntingLevels":awc_stunting_level.to_dict(orient="records"),
                    "stuntingClassification":awc_stunting_classification.to_dict(orient="records"),
                    "discrepancy":awc_level_disc.to_dict(orient="records"),
                }
            }
        except Exception as e:
            logger.error(f"Error preparing final response data: {e}")
            return (0, f"Error preparing final response data", [])

        return (1, "Success", response_data)

    except Exception as e:
        logger.error(f"An unexpected error occurred during data analysis: {e}. Please check the input data and column names.")
        # Catch any unexpected errors during the overall function execution
        return (0, f"An unexpected error occurred during data analysis. Please check the input data and column names.", [])
