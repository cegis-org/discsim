import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import base64
import json
from scipy.stats import percentileofscore

def error_handling(params):
    return (1, "Success")
    
def excel_percentrank_inc(series, value):
        if pd.isna(value) or value not in series.values:
            return 0  # Handle missing/irrelevant values (like Excel)
        ranked = series.rank(method="min")  # Match Excel's tie behavior
        count = len(series)
        return round((ranked[series == value].iloc[0] - 1) / (count - 1) * 100,1)

def anganwadi_center_data_anaylsis(file: pd.DataFrame):
    df = file
    required_columns = [
        'Id', 
        'Status_Wasting', 'Sup_Status_Wasting', 
        'Status_UW', 'Sup_Status_UW', 
        'Height', 'Sup_Height', 
        'Weight', 'Sup_Weight', 
        'Muac', 'Sup_Muac', 
        'AWC_ID', 'Sec_ID', 'Proj_Name', 'D_Name','AWC_Name'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            return (0, f"Error: Required column '{col}' is missing from the data.",[])

    # Mismatch Classification Conditions
    df['AWT_Normal_Sup_SAM'] = ((df['Sup_Status_Wasting'] == "SAM") & (df['Status_Wasting'] == "Normal")) * 1
    df['AWT_Normal_Sup_MAM'] = ((df['Sup_Status_Wasting'] == "MAM") & (df['Status_Wasting'] == "Normal")) * 1
    df['AWT_MAM_Sup_SAM'] = ((df['Sup_Status_Wasting'] == "SAM") & (df['Status_Wasting'] == "MAM")) * 1

    df['AWT_Normal_Sup_SUW'] = ((df['Sup_Status_UW'] == "SUW") & (df['Status_UW'] == "Normal")) * 1
    df['AWT_Normal_Sup_MUW'] = ((df['Sup_Status_UW'] == "MUW") & (df['Status_UW'] == "Normal")) * 1
    
    df['AWT_height_eq_Sup_height'] = (df['Height'] == df['Sup_Height']) * 1
    df['AWT_weight_eq_Sup_weight'] = (df['Weight'] == df['Sup_Weight']) * 1
    
    # % Mismatch Discrepancy Condition - If any of the above conditions are true
    df['Discrepancy'] = (
        df['AWT_Normal_Sup_SAM'] | df['AWT_Normal_Sup_MAM'] | df['AWT_MAM_Sup_SAM'] |
        df['AWT_Normal_Sup_SUW'] | df['AWT_Normal_Sup_MUW'] |
        (df['AWT_height_eq_Sup_height'] == 1) | (df['AWT_weight_eq_Sup_weight'] == 1)
    )

    # Wasting Conditions
    df['AWT_Wasting'] = ((df['Status_Wasting'] == "SAM") | (df['Status_Wasting'] == "MAM")) * 1
    df['Supervisor_Wasting'] = ((df['Sup_Status_Wasting'] == "SAM") | (df['Sup_Status_Wasting'] == "MAM")) * 1
    df['AWT_Sup_Same_Wasting'] = (df['Status_Wasting'] == df['Sup_Status_Wasting']) * 1
    df['AWT_Sup_Other_Misclassifications_Wasting'] = ((df['AWT_Sup_Same_Wasting'] == 0) & (df['AWT_Normal_Sup_SAM'] == 0) & (df['AWT_Normal_Sup_MAM'] == 0) & (df['AWT_MAM_Sup_SAM'] == 0)) * 1
    
    # Underweight Conditions
    df['AWT_Underweight'] = ((df['Status_UW'] == "SUW") | (df['Status_UW'] == "MUW")) * 1
    df['Supervisor_Underweight'] = ((df['Sup_Status_UW'] == "SUW") | (df['Sup_Status_UW'] == "MUW")) * 1
    df['AWT_Sup_Same_Underweight'] = (df['Status_UW'] == df['Sup_Status_UW']) * 1
    df['AWT_Sup_Other_Misclassifications_Underweight'] = ((df['AWT_Sup_Same_Underweight'] == 0) & (df['AWT_Normal_Sup_SUW'] == 0) & (df['AWT_Normal_Sup_MUW'] == 0)) * 1

    # Date Conversions
    df['WeightDate'] = pd.to_datetime(df['WeightDate'], format='%d/%m/%Y', errors='coerce')
    df['Sup_WeightDate'] = pd.to_datetime(df['Sup_WeightDate'], format='%d/%m/%Y', errors='coerce')
    df['Gap between AWT Sup Measurements'] = (df['Sup_WeightDate'] - df['WeightDate']).dt.days

    # Stunting Conditions
    df['AWT_Stunting'] = ((df['Status_Stunting'] == 'MAM') | (df['Status_Stunting'] == 'SAM')) * 1
    df['Supervisor_Stunting'] = ((df['Sup_Status_Stunting'] == 'MAM') | (df['Sup_Status_Stunting'] == 'SAM')) * 1
    df['AWT_Normal_Sup_Stunt_SAM'] = ((df['Status_Stunting'] == 'Normal') & (df['Sup_Status_Stunting'] == 'SAM')) * 1
    df['AWT_Normal_Sup_Stunt_MAM'] = ((df['Status_Stunting'] == 'Normal') & (df['Sup_Status_Stunting'] == 'MAM')) * 1
    df['AWT_MAM_Sup_Stunt_SAM'] = ((df['Status_Stunting'] == 'MAM') & (df['Sup_Status_Stunting'] == 'SAM')) * 1
    df['AWT_Sup_Same_Stunting'] = (df['Status_Stunting'] == df['Sup_Status_Stunting']) * 1
    df['AWT_Sup_Other_Misclassifications_Stunting'] =((df['AWT_Sup_Same_Stunting'] == 0) & (df['AWT_Normal_Sup_Stunt_MAM'] == 0) & (df['AWT_Normal_Sup_Stunt_SAM'] == 0)) * 1
    
    # Height-Weight Diff
    df['AWT_Sup_Height_Difference'] = np.round(np.where((df['Height'] != 0) & (df['Sup_Height'] != 0) & (df['Height'] != df['Sup_Height']),abs(df['Sup_Height'] - df['Height']), np.nan), 1)
    df['AWT_Sup_Weight_Difference'] = np.where((df['Weight'] != 0) & (df['Sup_Weight'] != 0) & (df['Weight'] != df['Sup_Weight']), abs(df['Weight'] - df['Sup_Weight']), np.nan)

    # Children Classifications
    df['0-3 years old'] = (df['AgeinMonthsAsDate'] < 36) * 1
    df['3-6 years old'] = np.where(df['0-3 years old'] == 0, 1, 0)

    numeric_cols = ["Height", "Weight", "Muac"]
    df['Height_Same'] = (df['Height'] == df['Sup_Height']) * 1
    df['Weight_Same'] = (df['Weight'] == df['Sup_Weight']) * 1
    df['Muac_Same'] = (df['Muac'] == df['Sup_Muac']) * 1
    for col in numeric_cols:
        df[f'{col}_Diff'] = abs(df[col] - df[f'Sup_{col}'])

    #Sample Size
    num_remeasurements = df.shape[0]

    #Exact Same Height and Weight - AW and Supervisor
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

    #Children Classifications
    children_category_data = pd.DataFrame({
        "Category": ["0-3 years old", "3-6 years old"],
        "Count": [
            df['0-3 years old'].sum(),
            df['3-6 years old'].sum()
        ],
        "Average Height Difference (cms)": [
            round(df.loc[df['0-3 years old'] == 1, 'AWT_Sup_Height_Difference'].mean(),1),
            round(df.loc[df['3-6 years old'] == 1, 'AWT_Sup_Height_Difference'].mean(),1)
        ],
        "Average Weight Difference (kgs)": [
            round(df.loc[df['0-3 years old'] == 1, 'AWT_Sup_Weight_Difference'].mean(),1),
            round(df.loc[df['3-6 years old'] == 1, 'AWT_Sup_Weight_Difference'].mean(),1)
        ],
        "Average gap in measurement (days)": [
            round(df.loc[df['0-3 years old'] == 1, 'Gap between AWT Sup Measurements'].mean(),0),
            round(df.loc[df['3-6 years old'] == 1, 'Gap between AWT Sup Measurements'].mean(),0)
        ]
    })
    
    #Wasting Levels
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

    #Wasting Classifications
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

    #UnderWeight Levels
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

    #Underweight Misclassification
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

    #Stunting Levels
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

    # Project Level Analysis
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

    #Project Level Discrepancy
    project_level_disc = df.groupby('Proj_Name').agg(
        Total_Remeasurements=('Proj_Name', 'count'),
        Discrepancy_remeasurements=('Discrepancy', 'sum'),
    ).reset_index()
    project_level_disc['Discrepancy Rate (%)'] = np.where(project_level_disc['Total_Remeasurements'] > 15,round((project_level_disc['Discrepancy_remeasurements'] / project_level_disc['Total_Remeasurements']) * 100,1),0)
    project_level_disc['Non-Discrepancy Rate (%)'] = np.where(project_level_disc['Total_Remeasurements'] > 15,round(100 - project_level_disc['Discrepancy Rate (%)'],1),0)
    valid_disc_rates = project_level_disc[project_level_disc['Discrepancy Rate (%)'] > 0]['Non-Discrepancy Rate (%)']
    project_level_disc["Percentile_Rank (%)"] = project_level_disc["Non-Discrepancy Rate (%)"].apply(
        lambda x: excel_percentrank_inc(valid_disc_rates, x) if x > 0 else 0
    )
    project_level_disc['Zone'] = np.where(
    project_level_disc['Discrepancy Rate (%)'] > 0,
    np.select(
        [
            project_level_disc['Percentile_Rank (%)'] >= 75, #green threshold
            project_level_disc['Percentile_Rank (%)'] <= 25 #red threshold
        ], ['Green', 'Red'],default='Yellow'),'')

    # Sector Level Analysis
    # Equal Same Height
    sector_analysis_eq_height = df.groupby('Sec_Name').agg(
        Total_Remeasurements=('Height', 'count'),
        Exact_Same_Height=('AWT_height_eq_Sup_height', 'sum')
    ).reset_index()
    sector_analysis_eq_height['Exact_Same_Height_%'] = np.where(sector_analysis_eq_height['Total_Remeasurements'] > 15,round((sector_analysis_eq_height['Exact_Same_Height'] / sector_analysis_eq_height['Total_Remeasurements']) * 100, 1),0)

    # Equal Same Weight
    sector_analysis_eq_weight = df.groupby('Sec_Name').agg(
        Total_Remeasurements=('Weight', 'count'),
        Exact_Same_Weight=('AWT_weight_eq_Sup_weight', 'sum')
    ).reset_index()
    sector_analysis_eq_weight['Exact_Same_Weight_%'] = np.where(sector_analysis_eq_weight['Total_Remeasurements'] > 15,round((sector_analysis_eq_weight['Exact_Same_Weight'] / sector_analysis_eq_weight['Total_Remeasurements']) * 100, 1),0) 
    #15 is case-threshold for user input (can be changed at each level)

    # Wasting Levels
    sector_analysis_wasting_levels = df.groupby('Sec_Name').agg(
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

    #Sector Level Discrepancy
    sector_level_disc = df.groupby('Sec_Name').agg(
        Total_Remeasurements=('Sec_Name', 'count'),
        Discrepancy_remeasurements=('Discrepancy', 'sum'),
    ).reset_index()
    sector_level_disc['Discrepancy Rate (%)'] = np.where(sector_level_disc['Total_Remeasurements'] > 15,round((sector_level_disc['Discrepancy_remeasurements'] / sector_level_disc['Total_Remeasurements']) * 100,1),0)
    sector_level_disc['Non-Discrepancy Rate (%)'] = np.where(sector_level_disc['Total_Remeasurements'] > 15,round(100 - sector_level_disc['Discrepancy Rate (%)'],1),0)
    valid_disc_rates = sector_level_disc[sector_level_disc['Discrepancy Rate (%)'] > 0]['Non-Discrepancy Rate (%)']
    sector_level_disc["Percentile_Rank (%)"] = sector_level_disc["Non-Discrepancy Rate (%)"].apply(
        lambda x: excel_percentrank_inc(valid_disc_rates, x) if x > 0 else 0
    )
    sector_level_disc['Zone'] = np.where(
    sector_level_disc['Discrepancy Rate (%)'] > 0,
    np.select(
        [
            sector_level_disc['Percentile_Rank (%)'] >= 80, #green threshold
            sector_level_disc['Percentile_Rank (%)'] <= 30 #red threshold
        ], ['Green', 'Red'],default='Yellow'),'')
    
    # AWC Level Analysis
    # Equal Same Height
    awc_analysis_eq_height = df.groupby('AWC_Name').agg(
        Total_Remeasurements=('Height', 'count'),
        Exact_Same_Height=('AWT_height_eq_Sup_height', 'sum')
    ).reset_index()
    awc_analysis_eq_height['Exact_Same_Height_%'] = np.where(awc_analysis_eq_height['Total_Remeasurements'] > 15,round((awc_analysis_eq_height['Exact_Same_Height'] / awc_analysis_eq_height['Total_Remeasurements']) * 100, 1),0)

    # Equal Same Weight
    awc_analysis_eq_weight = df.groupby('AWC_Name').agg(
        Total_Remeasurements=('Weight', 'count'),
        Exact_Same_Weight=('AWT_weight_eq_Sup_weight', 'sum')
    ).reset_index()
    awc_analysis_eq_weight['Exact_Same_Weight_%'] = np.where(awc_analysis_eq_weight['Total_Remeasurements'] > 15,round((awc_analysis_eq_weight['Exact_Same_Weight'] / awc_analysis_eq_weight['Total_Remeasurements']) * 100, 1),0) 
    #15 is case-threshold for user input (can be changed at each level)

    #Wasting Levels
    awc_analysis_wasting_levels = df.groupby('AWC_Name').agg(
        Total_Remeasurements=('AWC_Name', 'count'),
        AWT_SAM=('Status_Wasting', lambda x: (x == 'SAM').sum()),
        AWT_Wasting=('AWT_Wasting', 'sum'),
        Supervisor_SAM=('Sup_Status_Wasting', lambda x: (x == 'SAM').sum()),
        Supervisor_Wasting=('Supervisor_Wasting', 'sum')
    ).reset_index()
    awc_analysis_wasting_levels['AWT_SAM_%'] = round((awc_analysis_wasting_levels['AWT_SAM'] / awc_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
    awc_analysis_wasting_levels['AWT_Wasting_%'] = round((awc_analysis_wasting_levels['AWT_Wasting'] / awc_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
    awc_analysis_wasting_levels['Supervisor_SAM_%'] = round((awc_analysis_wasting_levels['Supervisor_SAM'] / awc_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
    awc_analysis_wasting_levels['Supervisor_Wasting_%'] = round((awc_analysis_wasting_levels['Supervisor_Wasting'] / awc_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)
    awc_analysis_wasting_levels['Sup-AWT_Difference_%'] = round(((awc_analysis_wasting_levels['Supervisor_Wasting'] - awc_analysis_wasting_levels['AWT_Wasting']) / awc_analysis_wasting_levels['Total_Remeasurements']) * 100, 0)

    # Wasting Classification
    awc_analysis_wasting_classification = df.groupby('AWC_Name').agg(
        Total_Remeasurements=('AWC_Name', 'count'),
        AWT_Normal_Sup_SAM=('AWT_Normal_Sup_SAM', 'sum'),
        AWT_Normal_Sup_MAM=('AWT_Normal_Sup_MAM', 'sum'),
        AWT_MAM_Sup_SAM=('AWT_MAM_Sup_SAM', 'sum'),
        Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Wasting', 'sum'),
        Same_Classifications=('AWT_Sup_Same_Wasting', 'sum'),
    ).reset_index()
    awc_analysis_wasting_classification['AWT_Normal_Sup_SAM_%'] = round((awc_analysis_wasting_classification['AWT_Normal_Sup_SAM'] / awc_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
    awc_analysis_wasting_classification['AWT_Normal_Sup_MAM_%'] = round((awc_analysis_wasting_classification['AWT_Normal_Sup_MAM'] / awc_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
    awc_analysis_wasting_classification['AWT_MAM_Sup_SAM_%'] = round((awc_analysis_wasting_classification['AWT_MAM_Sup_SAM'] / awc_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
    awc_analysis_wasting_classification['Other_Misclassifications_%'] = round((awc_analysis_wasting_classification['Other_Misclassifications'] / awc_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)
    awc_analysis_wasting_classification['Same_Classifications_%'] = round((awc_analysis_wasting_classification['Same_Classifications'] / awc_analysis_wasting_classification['Total_Remeasurements']) * 100, 0)

    # Underweight Levels
    awc_analysis_uw_levels = df.groupby('AWC_Name').agg(
        Total_Remeasurements=('AWC_Name', 'count'),
        AWT_SUW=('Status_UW', lambda x: (x == 'SUW').sum()),
        Sup_SUW=('Sup_Status_UW', lambda x: (x == 'SUW').sum()),
        AWT_Underweight=('AWT_Underweight', 'sum'),
        Sup_Underweight=('Supervisor_Underweight', 'sum')
    ).reset_index()
    awc_analysis_uw_levels['AWT_SUW_%'] = round((awc_analysis_uw_levels['AWT_SUW'] / awc_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
    awc_analysis_uw_levels['Sup_SUW_%'] = round((awc_analysis_uw_levels['Sup_SUW'] / awc_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
    awc_analysis_uw_levels['AWT_Underweight_%'] = round((awc_analysis_uw_levels['AWT_Underweight'] / awc_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
    awc_analysis_uw_levels['Sup_Underweight_%'] = round((awc_analysis_uw_levels['Sup_Underweight'] / awc_analysis_uw_levels['Total_Remeasurements']) * 100, 0)
    awc_analysis_uw_levels['Sup-AWT_Difference_%'] = round(((awc_analysis_uw_levels['Sup_Underweight'] - awc_analysis_uw_levels['AWT_Underweight']) / awc_analysis_uw_levels['Total_Remeasurements']) * 100, 0)

    # Underweight Classification
    awc_analysis_underweight_classification = df.groupby('AWC_Name').agg(
        Total_Remeasurements=('AWC_Name', 'count'),
        AWT_Normal_Sup_SUW=('AWT_Normal_Sup_SUW', 'sum'),
        AWT_Normal_Sup_MUW=('AWT_Normal_Sup_MUW', 'sum'),
        Other_Misclassifications=('AWT_Sup_Other_Misclassifications_Underweight', 'sum'),
        Same_Classifications=('AWT_Sup_Same_Underweight', 'sum'),
    ).reset_index()
    awc_analysis_underweight_classification['AWT_Normal_Sup_SUW_%'] = round((awc_analysis_underweight_classification['AWT_Normal_Sup_SUW'] / awc_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
    awc_analysis_underweight_classification['AWT_Normal_Sup_MUW_%'] = round((awc_analysis_underweight_classification['AWT_Normal_Sup_MUW'] / awc_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
    awc_analysis_underweight_classification['Other_Misclassifications_%'] = round((awc_analysis_underweight_classification['Other_Misclassifications'] / awc_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)
    awc_analysis_underweight_classification['Same_Classifications_%'] = round((awc_analysis_underweight_classification['Same_Classifications'] / awc_analysis_underweight_classification['Total_Remeasurements']) * 100, 0)

    #AWC Level Discrepancy
    awc_level_disc = df.groupby('AWC_Name').agg(
        Total_Remeasurements=('AWC_Name', 'count'),
        Discrepancy_remeasurements=('Discrepancy', 'sum'),
    ).reset_index()
    awc_level_disc['Discrepancy Rate (%)'] = np.where(awc_level_disc['Total_Remeasurements'] > 5,round((awc_level_disc['Discrepancy_remeasurements'] / awc_level_disc['Total_Remeasurements']) * 100,1),0)
    awc_level_disc['Non-Discrepancy Rate (%)'] = np.where(awc_level_disc['Total_Remeasurements'] > 5,round(100 - awc_level_disc['Discrepancy Rate (%)'],1),0)
    valid_disc_rates = awc_level_disc[awc_level_disc['Discrepancy Rate (%)'] > 0]['Non-Discrepancy Rate (%)']
    awc_level_disc["Percentile_Rank (%)"] = awc_level_disc["Non-Discrepancy Rate (%)"].apply(
        lambda x: excel_percentrank_inc(valid_disc_rates, x) if x > 0 else 0
    )
    awc_level_disc['Zone'] = np.where(
    awc_level_disc['Discrepancy Rate (%)'] > 0,
    np.select(
        [
            awc_level_disc['Percentile_Rank (%)'] >= 80, #green threshold
            awc_level_disc['Percentile_Rank (%)'] <= 30 #red threshold
        ], ['Green', 'Red'],default='Yellow'),'')

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
            "childrenCategory":children_category_data.to_dict(orient="records"),
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
            "discrepancy":project_level_disc.to_dict(orient="records"),
        },
        "sectorLevelInsights":{
            "sameHeight":sector_analysis_eq_height.to_dict(orient="records"),
            "sameWeight":sector_analysis_eq_weight.to_dict(orient="records"),
            "wastingLevels":sector_analysis_wasting_levels.to_dict(orient="records"),
            "wastingClassification":sector_analysis_wasting_classification.to_dict(orient="records"),
            "underweightLevels":sector_analysis_uw_levels.to_dict(orient="records"),
            "underweightClassification":sector_analysis_underweight_classification.to_dict(orient="records"),
            "discrepancy":sector_level_disc.to_dict(orient="records"),
        },
        "awcLevelInsights":{
            "sameHeight":awc_analysis_eq_height.to_dict(orient="records"),
            "sameWeight":awc_analysis_eq_weight.to_dict(orient="records"),
            "wastingLevels":awc_analysis_wasting_levels.to_dict(orient="records"),
            "wastingClassification":awc_analysis_wasting_classification.to_dict(orient="records"),
            "underweightLevels":awc_analysis_uw_levels.to_dict(orient="records"),
            "underweightClassification":awc_analysis_underweight_classification.to_dict(orient="records"),
            "discrepancy":awc_level_disc.to_dict(orient="records"),
        }
    }
    
    return (1, "Success",response_data)
