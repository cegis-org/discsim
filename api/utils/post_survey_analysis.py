import pandas as pd
from typing import Dict, Any, List
import plotly.express as px

def calculate_discrepancy_scores(df: pd.DataFrame, margin_of_error_height: float = 0.0, margin_of_error_weight: float = 0.0) -> Dict[str, Any]:
    """
    Calculate discrepancy measures and composite discrepancy score for each L0 and L1 combination.
    Additionally, generate interactive Plotly plots for these metrics.

    Args:
        df (pd.DataFrame): DataFrame containing the survey data.
        margin_of_error_height (float): Acceptable margin of error for height measurements.
        margin_of_error_weight (float): Acceptable margin of error for weight measurements.

    Returns:
        Dict[str, Any]: Dictionary containing the results per L0 and L1, including plots.
    """
    
    # Ensure necessary columns are present
    required_columns = [
        'child', 'L0_height', 'L1_height', 'L0_weight', 'L1_weight', 
        'L0_id', 'L0_name', 'L1_id', 'L1_name', 
        'wasting_L0', 'stunting_L0', 'underweight_L0', 
        'wasting_L1', 'stunting_L1', 'underweight_L1'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the data.")
    
    # Calculate discrepancies
    df['height_discrepancy_cm'] = (df['L1_height'] - df['L0_height']).abs() - margin_of_error_height
    df['weight_discrepancy_kg'] = (df['L1_weight'] - df['L0_weight']).abs() - margin_of_error_weight
    
    # Clip negative discrepancies to zero
    df['height_discrepancy_cm'] = df['height_discrepancy_cm'].clip(lower=0)
    df['weight_discrepancy_kg'] = df['weight_discrepancy_kg'].clip(lower=0)
    
    # Classification accuracy metrics
    df['wasting_accuracy'] = df.apply(lambda row: 'Accurate' if row['wasting_L0'] == row['wasting_L1'] else 'Misclassified', axis=1)
    df['stunting_accuracy'] = df.apply(lambda row: 'Accurate' if row['stunting_L0'] == row['stunting_L1'] else 'Misclassified', axis=1)
    df['underweight_accuracy'] = df.apply(lambda row: 'Accurate' if row['underweight_L0'] == row['underweight_L1'] else 'Misclassified', axis=1)
    
    # Group by L0_id and L1_id
    grouped = df.groupby(['L0_id', 'L1_id'])
    
    results: List[Dict[str, Any]] = []
    
    for (l0_id, l1_id), group in grouped:
        l0_name = group['L0_name'].iloc[0]
        l1_name = group['L1_name'].iloc[0]
        
        # Discrepancy Measures
        avg_height_discrepancy = group['height_discrepancy_cm'].mean()
        avg_weight_discrepancy = group['weight_discrepancy_kg'].mean()
        
        height_discrepancy_prevalence = (group['height_discrepancy_cm'] > 0).mean() * 100
        weight_discrepancy_prevalence = (group['weight_discrepancy_kg'] > 0).mean() * 100
        
        # Measurement Accuracy
        height_accuracy = (group['height_discrepancy_cm'] <= margin_of_error_height).mean() * 100
        weight_accuracy = (group['weight_discrepancy_kg'] <= margin_of_error_weight).mean() * 100
        
        # Classification Accuracy - Wasting
        wasting_acc = (group['wasting_accuracy'] == 'Accurate').mean() * 100
        wasting_mam_as_normal = group[(group['wasting_L1'].isin(['MAM', 'SAM'])) & (group['wasting_L0'] == 'Normal')].shape[0] / group.shape[0] * 100
        wasting_sam_as_mam = group[(group['wasting_L1'] == 'SAM') & (group['wasting_L0'] == 'MAM')].shape[0] / group.shape[0] * 100
        wasting_other_misclassification = 100 - (wasting_acc + wasting_mam_as_normal + wasting_sam_as_mam)
        
        # Classification Accuracy - Stunting
        stunting_acc = (group['stunting_accuracy'] == 'Accurate').mean() * 100
        stunting_mam_as_normal = group[(group['stunting_L1'].isin(['MAM', 'SAM'])) & (group['stunting_L0'] == 'Normal')].shape[0] / group.shape[0] * 100
        stunting_sam_as_mam = group[(group['stunting_L1'] == 'SAM') & (group['stunting_L0'] == 'MAM')].shape[0] / group.shape[0] * 100
        stunting_other_misclassification = 100 - (stunting_acc + stunting_mam_as_normal + stunting_sam_as_mam)
        
        # Classification Accuracy - Underweight
        underweight_acc = (group['underweight_accuracy'] == 'Accurate').mean() * 100
        underweight_mam_as_normal = group[(group['underweight_L1'].isin(['MAM', 'SAM'])) & (group['underweight_L0'] == 'Normal')].shape[0] / group.shape[0] * 100
        underweight_sam_as_mam = group[(group['underweight_L1'] == 'SAM') & (group['underweight_L0'] == 'MAM')].shape[0] / group.shape[0] * 100
        underweight_other_misclassification = 100 - (underweight_acc + underweight_mam_as_normal + underweight_sam_as_mam)
        
        # Composite Score Calculation
        # Normalize measures (Higher values indicate worse performance)
        max_height_discrepancy = df['height_discrepancy_cm'].max()
        max_weight_discrepancy = df['weight_discrepancy_kg'].max()
        
        norm_height_discrepancy = avg_height_discrepancy / max_height_discrepancy if max_height_discrepancy != 0 else 0
        norm_weight_discrepancy = avg_weight_discrepancy / max_weight_discrepancy if max_weight_discrepancy != 0 else 0
        
        # Invert accuracies to reflect higher values as worse performance
        norm_height_accuracy = 1 - (height_accuracy / 100)
        norm_weight_accuracy = 1 - (weight_accuracy / 100)
        norm_wasting_accuracy = 1 - (wasting_acc / 100)
        
        # Composite Score Calculation with Equal Weights
        weights = {
            'average_height_discrepancy': 1,
            'average_weight_discrepancy': 1,
            'height_accuracy': 1,
            'weight_accuracy': 1,
            'classification_accuracy_wasting': 1
        }
        
        composite_score = (
            weights['average_height_discrepancy'] * norm_height_discrepancy +
            weights['average_weight_discrepancy'] * norm_weight_discrepancy +
            weights['height_accuracy'] * norm_height_accuracy +
            weights['weight_accuracy'] * norm_weight_accuracy +
            weights['classification_accuracy_wasting'] * norm_wasting_accuracy
        )
        
        # Normalize composite score to 0-100 scale
        max_weight_sum = sum(weights.values())
        composite_score = (composite_score / max_weight_sum) * 100
        
        # Append results with casting to native Python types
        results.append({
            'L0_id': str(l0_id),  # Keep as string to handle alphanumerics
            'L0_name': str(l0_name),
            'L1_id': str(l1_id),  # Keep as string to handle alphanumerics
            'L1_name': str(l1_name),
            'average_height_discrepancy_cm': float(round(avg_height_discrepancy, 2)),
            'average_weight_discrepancy_kg': float(round(avg_weight_discrepancy, 2)),
            'height_discrepancy_prevalence_percent': float(round(height_discrepancy_prevalence, 2)),
            'weight_discrepancy_prevalence_percent': float(round(weight_discrepancy_prevalence, 2)),
            'height_accuracy_percent': float(round(height_accuracy, 2)),
            'weight_accuracy_percent': float(round(weight_accuracy, 2)),
            'classification_accuracy_wasting_percent': float(round(wasting_acc, 2)),
            'classification_mam_as_normal_percent': float(round(wasting_mam_as_normal, 2)),
            'classification_sam_as_mam_percent': float(round(wasting_sam_as_mam, 2)),
            'classification_other_wasting_misclassification_percent': float(round(wasting_other_misclassification, 2)),
            'classification_accuracy_stunting_percent': float(round(stunting_acc, 2)),
            'classification_mam_as_normal_stunting_percent': float(round(stunting_mam_as_normal, 2)),
            'classification_sam_as_mam_stunting_percent': float(round(stunting_sam_as_mam, 2)),
            'classification_other_stunting_misclassification_percent': float(round(stunting_other_misclassification, 2)),
            'classification_accuracy_underweight_percent': float(round(underweight_acc, 2)),
            'classification_mam_as_normal_underweight_percent': float(round(underweight_mam_as_normal, 2)),
            'classification_sam_as_mam_underweight_percent': float(round(underweight_sam_as_mam, 2)),
            'classification_other_underweight_misclassification_percent': float(round(underweight_other_misclassification, 2)),
            'composite_discrepancy_score': float(round(composite_score, 2))
        })
    
    # After gathering all results, generate Plotly plots
    discrepancy_df = pd.DataFrame(results)
    
    plots = {}
    
    # Fix SettingWithCopyWarning by making copies
    discrepancy_df = discrepancy_df.copy()
    
    # Plot 1: Average Height Discrepancy (cm) per L0 (Horizontal Bar)
    discrepancy_df_sorted = discrepancy_df.sort_values('average_height_discrepancy_cm', ascending=False)
    fig_height = px.bar(
        discrepancy_df_sorted,
        x='average_height_discrepancy_cm',
        y='L0_name',
        orientation='h',
        title='Average Height Discrepancy (cm)',
        labels={'L0_name': 'L0 Name', 'average_height_discrepancy_cm': 'Average Height Discrepancy (cm)'},
        color='average_height_discrepancy_cm',
        color_continuous_scale='Viridis',
        text='average_height_discrepancy_cm'  
    )
    fig_height.update_traces(
        texttemplate='%{text:.2f}',
        textposition='inside'
    )
    fig_height.update_layout(
        yaxis={'categoryorder':'total ascending'},
        coloraxis_colorbar=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5,
            title='Average Height Discrepancy (cm)'
        )
    )
    plots['height_discrepancy_plot'] = fig_height.to_json()
    
    # Plot 2: Average Weight Discrepancy (kg) per L0 (Horizontal Bar)
    discrepancy_df_sorted = discrepancy_df.sort_values('average_weight_discrepancy_kg', ascending=False)
    fig_weight = px.bar(
        discrepancy_df_sorted,
        x='average_weight_discrepancy_kg',
        y='L0_name',
        orientation='h',
        title='Average Weight Discrepancy (kg)',
        labels={'L0_name': 'L0 Name', 'average_weight_discrepancy_kg': 'Average Weight Discrepancy (kg)'},
        color='average_weight_discrepancy_kg',
        color_continuous_scale='Magma',
        text='average_weight_discrepancy_kg'  
    )
    fig_weight.update_traces(
        texttemplate='%{text:.2f}',
        textposition='inside'
    )
    fig_weight.update_layout(
        yaxis={'categoryorder':'total ascending'},
        coloraxis_colorbar=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5,
            title='Average Weight Discrepancy (kg)'
        )
    )
    plots['weight_discrepancy_plot'] = fig_weight.to_json()
    
    # Plot: Combined Average Height and Weight Discrepancy per L0
    # Melt the data to long format
    discrepancy_melted = discrepancy_df[['L0_name', 'average_height_discrepancy_cm', 'average_weight_discrepancy_kg']].copy()
    discrepancy_melted = discrepancy_melted.melt(
        id_vars='L0_name',
        value_vars=['average_height_discrepancy_cm', 'average_weight_discrepancy_kg'],
        var_name='Measurement',
        value_name='Average Discrepancy'
    )
    # Rename Measurement values for better labels
    discrepancy_melted['Measurement'] = discrepancy_melted['Measurement'].replace({
        'average_height_discrepancy_cm': 'Height (cm)',
        'average_weight_discrepancy_kg': 'Weight (kg)'
    })
    # Sort the data
    discrepancy_melted.sort_values('Average Discrepancy', ascending=False, inplace=True)
    
    fig_combined_discrepancy = px.bar(
        discrepancy_melted,
        x='Average Discrepancy',
        y='L0_name',
        orientation='h',
        color='Measurement',
        barmode='group',
        title='Combined Average Height and Weight Discrepancy',
        labels={'L0_name': 'L0 Name', 'Average Discrepancy': 'Average Discrepancy'},
        text='Average Discrepancy'
    )
    fig_combined_discrepancy.update_traces(
        texttemplate='%{text:.2f}',
        textposition='inside'
    )
    fig_combined_discrepancy.update_layout(
        yaxis={'categoryorder':'total ascending'},
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.25,
            xanchor='center',
            x=0.5
        )
    )
    plots['combined_discrepancy_plot'] = fig_combined_discrepancy.to_json()
    
    # Plot 3: Height Measurement Accuracy (%) per L0 (Horizontal Bar)
    discrepancy_df_sorted = discrepancy_df.sort_values('height_accuracy_percent', ascending=False)
    fig_height_acc = px.bar(
        discrepancy_df_sorted,
        x='height_accuracy_percent',
        y='L0_name',
        orientation='h',
        title='Height Measurement Accuracy (%)',
        labels={'L0_name': 'L0 Name', 'height_accuracy_percent': 'Height Measurement Accuracy (%)'},
        color='height_accuracy_percent',
        color_continuous_scale='RdBu',
        text='height_accuracy_percent'  
    )
    fig_height_acc.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='inside'
    )
    fig_height_acc.update_layout(
        yaxis={'categoryorder':'total ascending'},
        coloraxis_colorbar=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5,
            title='Height Measurement Accuracy (%)'
        )
    )
    plots['height_accuracy_plot'] = fig_height_acc.to_json()
    
    # Plot 4: Weight Measurement Accuracy (%) per L0 (Horizontal Bar)
    discrepancy_df_sorted = discrepancy_df.sort_values('weight_accuracy_percent', ascending=False)
    fig_weight_acc = px.bar(
        discrepancy_df_sorted,
        x='weight_accuracy_percent',
        y='L0_name',
        orientation='h',
        title='Weight Measurement Accuracy (%)',
        labels={'L0_name': 'L0 Name', 'weight_accuracy_percent': 'Weight Measurement Accuracy (%)'},
        color='weight_accuracy_percent',
        color_continuous_scale='Plasma',
        text='weight_accuracy_percent'  
    )
    fig_weight_acc.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='inside'
    )
    fig_weight_acc.update_layout(
        yaxis={'categoryorder':'total ascending'},
        coloraxis_colorbar=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5,
            title='Weight Measurement Accuracy (%)'
        )
    )
    plots['weight_accuracy_plot'] = fig_weight_acc.to_json()
    
    # Plot 5: Classification Accuracy - Wasting vs L0 (Stacked Horizontal Bar)
    classification_wasting_df = discrepancy_df[[
        'L0_name', 
        'classification_accuracy_wasting_percent', 
        'classification_mam_as_normal_percent', 
        'classification_sam_as_mam_percent', 
        'classification_other_wasting_misclassification_percent'
    ]].copy()
    classification_wasting_df.rename(columns={
        'classification_accuracy_wasting_percent': 'Accurate',
        'classification_mam_as_normal_percent': 'MAM as Normal',
        'classification_sam_as_mam_percent': 'SAM as MAM',
        'classification_other_wasting_misclassification_percent': 'Other Misclassifications'
    }, inplace=True)
    classification_wasting_melted = classification_wasting_df.melt(
        id_vars='L0_name', 
        var_name='Classification',
        value_name='Percentage'
    )

    # Define color mapping
    color_map_wasting = {
        'Accurate': 'green',
        'MAM as Normal': 'darkred',
        'SAM as MAM': 'indigo',
        'Other Misclassifications': 'darkblue'
    }

    fig_class_wasting = px.bar(
        classification_wasting_melted,
        x='Percentage',
        y='L0_name',
        color='Classification',
        orientation='h',
        title='Classification Accuracy - Wasting vs L0',
        labels={'L0_name': 'L0 Name', 'Percentage': 'Percentage (%)'},
        text='Percentage',
        color_discrete_map=color_map_wasting
    )
    fig_class_wasting.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='inside'
    )
    fig_class_wasting.update_layout(
        barmode='stack',
        yaxis={'categoryorder': 'total ascending'},
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    plots['classification_wasting_plot'] = fig_class_wasting.to_json()

    
    # Plot 6: Classification Accuracy - Stunting vs L0 (Stacked Horizontal Bar)
    classification_stunting_df = discrepancy_df[[
        'L0_name', 
        'classification_accuracy_stunting_percent', 
        'classification_mam_as_normal_stunting_percent', 
        'classification_sam_as_mam_stunting_percent', 
        'classification_other_stunting_misclassification_percent'
    ]].copy()
    classification_stunting_df.rename(columns={
        'classification_accuracy_stunting_percent': 'Accurate',
        'classification_mam_as_normal_stunting_percent': 'MAM as Normal',
        'classification_sam_as_mam_stunting_percent': 'SAM as MAM',
        'classification_other_stunting_misclassification_percent': 'Other Misclassifications'
    }, inplace=True)
    classification_stunting_melted = classification_stunting_df.melt(
        id_vars='L0_name', 
        var_name='Classification',
        value_name='Percentage'
    )

    # Define color mapping
    color_map_stunting = {
        'Accurate': 'green',
        'MAM as Normal': 'darkred',
        'SAM as MAM': 'darkmagenta',
        'Other Misclassifications': 'navy'
    }

    fig_class_stunting = px.bar(
        classification_stunting_melted,
        x='Percentage',
        y='L0_name',
        color='Classification',
        orientation='h',
        title='Classification Accuracy - Stunting vs L0',
        labels={'L0_name': 'L0 Name', 'Percentage': 'Percentage (%)'},
        text='Percentage',
        color_discrete_map=color_map_stunting
    )
    fig_class_stunting.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='inside'
    )
    fig_class_stunting.update_layout(
        barmode='stack',
        yaxis={'categoryorder': 'total ascending'},
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    plots['classification_stunting_plot'] = fig_class_stunting.to_json()

    
    # Plot 7: Classification Accuracy - Underweight vs L0 (Stacked Horizontal Bar)
    classification_underweight_df = discrepancy_df[[
        'L0_name', 
        'classification_accuracy_underweight_percent', 
        'classification_mam_as_normal_underweight_percent', 
        'classification_sam_as_mam_underweight_percent', 
        'classification_other_underweight_misclassification_percent'
    ]].copy()
    classification_underweight_df.rename(columns={
        'classification_accuracy_underweight_percent': 'Accurate',
        'classification_mam_as_normal_underweight_percent': 'MAM as Normal',
        'classification_sam_as_mam_underweight_percent': 'SAM as MAM',
        'classification_other_underweight_misclassification_percent': 'Other Misclassifications'
    }, inplace=True)
    classification_underweight_melted = classification_underweight_df.melt(
        id_vars='L0_name', 
        var_name='Classification',
        value_name='Percentage'
    )

    # Define color mapping
    color_map_underweight = {
        'Accurate': 'green',
        'MAM as Normal': 'darkred',
        'SAM as MAM': 'purple',
        'Other Misclassifications': 'brown'
    }

    fig_class_underweight = px.bar(
        classification_underweight_melted,
        x='Percentage',
        y='L0_name',
        color='Classification',
        orientation='h',
        title='Classification Accuracy - Underweight vs L0',
        labels={'L0_name': 'L0 Name', 'Percentage': 'Percentage (%)'},
        text='Percentage',
        color_discrete_map=color_map_underweight
    )
    fig_class_underweight.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='inside'
    )
    fig_class_underweight.update_layout(
        barmode='stack',
        yaxis={'categoryorder': 'total ascending'},
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    plots['classification_underweight_plot'] = fig_class_underweight.to_json()

    
    # Plot 8: Composite Discrepancy Score per L0 (Horizontal Bar)
    discrepancy_df_sorted = discrepancy_df.sort_values('composite_discrepancy_score', ascending=False)
    fig_composite = px.bar(
        discrepancy_df_sorted,
        x='composite_discrepancy_score',
        y='L0_name',
        orientation='h',
        title='Composite Discrepancy Score per L0',
        labels={'L0_name': 'L0 Name', 'composite_discrepancy_score': 'Composite Discrepancy Score'},
        color='composite_discrepancy_score',
        color_continuous_scale='Cividis',
        text='composite_discrepancy_score'  
    )
    fig_composite.update_traces(
        texttemplate='%{text:.2f}',
        textposition='inside'
    )
    fig_composite.update_layout(
        yaxis={'categoryorder':'total ascending'},
        coloraxis_colorbar=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5,
            title='Composite Discrepancy Score'
        )
    )
    plots['composite_discrepancy_plot'] = fig_composite.to_json()
    
    return {
        'grouped_discrepancy_scores': results,
        'plots': plots
    }
