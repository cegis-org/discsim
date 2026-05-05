import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .generate_ecd_dummy_data import generate_nested_distortion_parameters
from .generate_ecd_dummy_data import generate_nested_measurements
from .generate_ecd_dummy_data import get_L1_L2_pairwise_data

def calculate_discrepancy_scores(measurements1, measurements2, variable, method, 
                               make_plot=False, plot_title=None,
                               measurements1_name=None, measurements2_name=None,
                               discrepancy_unit=None):
    """Calculate discrepancy scores between two sets of measurements for a given variable.
    
    Args:
        measurements1 (dict): First set of measurements
        measurements2 (dict): Second set of measurements 
        variable (str): Variable to compare ('height', 'weight', 'haz', 'waz' or 'whz')
        method (str): Method for calculating discrepancy:
            - 'percent_difference'
            - 'absolute_difference'
            - 'absolute_percent_difference'
            - 'simple_difference'
            - 'percent_non_match'
            - 'directional_percent_non_match'
            - 'directional_difference'
        make_plot (bool): Whether to create visualization plots
        plot_title (str): Title for the plots
        measurements1_name (str): Label for measurements1 on x-axis
        measurements2_name (str): Label for measurements2 on x-axis
        discrepancy_unit (str): Unit for discrepancy scores (e.g., 'kg' for weight)
    
    Returns:
        numpy.array: Array of discrepancy scores for each measurement
    """
    
    # Check if variable exists in both dictionaries
    if variable not in measurements1 or variable not in measurements2:
        raise KeyError(f"Variable '{variable}' must be present in both measurement dictionaries")
        
    # Get the measurements for the specified variable
    values1 = measurements1[variable]
    values2 = measurements2[variable]
    
    # Calculate individual discrepancy scores
    if method == "percent_difference":
        disc_scores = (values1 - values2) / values2 * 100
    elif method == "absolute_difference":
        disc_scores = np.abs(values1 - values2)
    elif method == "absolute_percent_difference":
        disc_scores = np.abs((values1 - values2) / values2 * 100)
    elif method == "simple_difference":
        disc_scores = values1 - values2
    else:
        raise ValueError("Method must be one of: percent_difference, absolute_difference, absolute_percent_difference, simple_difference")
    
    if make_plot:
        # Set default labels if not provided
        if measurements1_name is None:
            measurements1_name = "Measurements 1"
        if measurements2_name is None:
            measurements2_name = "Measurements 2"
        if plot_title is None:
            plot_title = f"Discrepancy Analysis for {variable}"
        if discrepancy_unit is None:
            discrepancy_label = "Discrepancy"
        else:
            discrepancy_label = f"Discrepancy ({discrepancy_unit})"
            
        # Create figure with 1 row and 3 columns
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(plot_title, fontsize=14)
        
        # Histogram of discrepancy scores
        ax1.hist(disc_scores, bins=50, color='lightgray', edgecolor='black')
        ax1.set_xlabel(discrepancy_label, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        
        # Scatter plot against measurements1
        ax2.scatter(values1, disc_scores, alpha=0.2, color='turquoise', s=20, linewidth=0)
        ax2.set_xlabel(f"{measurements1_name} {variable}", fontsize=12)
        ax2.set_ylabel('Unit-wise discrepancy', fontsize=12)
        
        # Scatter plot against measurements2
        ax3.scatter(values2, disc_scores, alpha=0.2, color='salmon', s=20, linewidth=0)
        ax3.set_xlabel(f"{measurements2_name} {variable}", fontsize=12)
        ax3.set_ylabel('Unit-wise discrepancy', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    return disc_scores

def calculate_ranks_L0_units(nested_measurements, measurement_var, method='simple_difference'):
    """
    Calculate real and measured ranks for units of L0s based on average discrepancy scores.
    
    Args:
        nested_measurements (dict): Nested dictionary containing measurements structured as:
            L1_id -> L0_id -> measurement_type -> measurements
        measurement_var (str): Variable to calculate discrepancy scores for (e.g., 'weight', 'height')
        method (str): Method to calculate discrepancy scores. Default: 'simple_difference'
            
    Returns:
        tuple: (real_ranks, measured_ranks) where each is an array of L1 unit ranks (1-based)
        ordered by the L1 unit IDs
    """
    # Store average discrepancy scores for each L1 unit
    unit_real_discrepancies = {}
    unit_measured_discrepancies = {}
    
    # Calculate average discrepancy scores for each L1 unit
    for L1_id in nested_measurements:
        if L1_id == 'metadata':
            continue
            
        # Store discrepancy scores for all L0s in this L1 unit
        real_discrepancies = []
        measured_discrepancies = []
        
        # Calculate discrepancy scores for each L0 in this L1 unit
        for L0_id in nested_measurements[L1_id]:
            if L0_id == 'L1_info':
                continue
                
            # Get measurements
            real_meas = nested_measurements[L1_id][L0_id]['real']
            L0_meas = nested_measurements[L1_id][L0_id]['L0']
            L1_meas = nested_measurements[L1_id][L0_id]['L1']
            
            # Calculate real discrepancy (L0 vs real)
            real_disc = calculate_discrepancy_scores(
                L0_meas['data'],
                real_meas['data'],
                measurement_var,
                method,
                make_plot=False
            )
            real_discrepancies.append(abs(real_disc.mean()))
            
            # Calculate measured discrepancy (L0 vs L1)
            # Only use children measured by L1
            L1_indices = L1_meas['data'].index
            L0_subset = {
                'data': L0_meas['data'].loc[L1_indices].copy(),
                'metadata': L0_meas['metadata'].copy()
            }
            
            measured_disc = calculate_discrepancy_scores(
                L0_subset['data'],
                L1_meas['data'],
                measurement_var,
                method,
                make_plot=False
            )
            measured_discrepancies.append(abs(measured_disc.mean()))
        
        # Calculate average discrepancy for this L1 unit
        unit_real_discrepancies[L1_id] = real_discrepancies
        unit_measured_discrepancies[L1_id] = measured_discrepancies

    # Convert to arrays ordered by L1_id
    L1_ids = sorted([L1_id for L1_id in unit_real_discrepancies.keys()])
    mean_real_discrepancies = np.array([np.mean(unit_real_discrepancies[L1_id]) for L1_id in L1_ids])
    mean_measured_discrepancies = np.array([np.mean(unit_measured_discrepancies[L1_id]) for L1_id in L1_ids])

    # Calculate ranks (1-based ranking, ascending order of discrepancy)
    real_ranks = mean_real_discrepancies.argsort().argsort() + 1
    measured_ranks = mean_measured_discrepancies.argsort().argsort() + 1

    return real_ranks, measured_ranks, unit_real_discrepancies, unit_measured_discrepancies

def calculate_ranks_L0s(nested_measurements, measurement_var, method='simple_difference'):
    """
    Calculate real and measured ranks for L0s within each L1 unit based on discrepancy scores.
    
    Args:
        nested_measurements (dict): Nested dictionary containing measurements structured as:
            L1_id -> L0_id -> measurement_type -> measurements
        measurement_var (str): Variable to calculate discrepancy scores for (e.g., 'weight', 'height')
        method (str): Method to calculate discrepancy scores. Default: 'simple_difference'
            
    Returns:
        dict: Dictionary with structure {L1_id: {'real_ranks': array, 'measured_ranks': array}}
              where ranks are 1-based and in ascending order of discrepancy
    """
    L0_ranks = {}
    
    # Calculate ranks for L0s within each L1 unit
    for L1_id in nested_measurements:
        if L1_id == 'metadata':
            continue
        
        # Store discrepancy scores for all L0s in this L1 unit
        real_discrepancies = []
        measured_discrepancies = []
        L0_ids = []
        
        # Calculate discrepancy scores for each L0 in this L1 unit
        for L0_id in nested_measurements[L1_id]:
            if L0_id == 'L1_info':
                continue
            
            L0_ids.append(L0_id)
            
            # Get measurements
            real_meas = nested_measurements[L1_id][L0_id]['real']
            L0_meas = nested_measurements[L1_id][L0_id]['L0']
            L1_meas = nested_measurements[L1_id][L0_id]['L1']
            
            # Calculate real discrepancy (L0 vs real)
            real_disc = calculate_discrepancy_scores(
                L0_meas['data'],
                real_meas['data'],
                measurement_var,
                method,
                make_plot=False
            )
            real_discrepancies.append(abs(real_disc.mean()))
            
            # Calculate measured discrepancy (L0 vs L1)
            # Only use children measured by L1
            L1_indices = L1_meas['data'].index
            L0_subset = L0_meas['data'].loc[L1_indices].copy()
            
            measured_disc = calculate_discrepancy_scores(
                L0_subset,
                L1_meas['data'],
                measurement_var,
                method,
                make_plot=False
            )
            measured_discrepancies.append(abs(measured_disc.mean()))
        
        # Convert to arrays
        real_discrepancies = np.array(real_discrepancies)
        measured_discrepancies = np.array(measured_discrepancies)
        
        # Calculate ranks (1-based ranking, ascending order of discrepancy)
        real_ranks = real_discrepancies.argsort().argsort() + 1
        measured_ranks = measured_discrepancies.argsort().argsort() + 1
        
        # Store ranks for this L1 unit
        L0_ranks[L1_id] = {
            'real_ranks': real_ranks,
            'measured_ranks': measured_ranks,
            'L0_ids': L0_ids
        }
    
    return L0_ranks

def plot_nested_measurements_and_ranks(nested_measurements, measurement_var, measurement_unit, 
                                     real_discrepancies, measured_discrepancies,
                                     real_ranks, measured_ranks, figsize=(15, 6)):
    """
    Visualize nested measurements and ranks using scatter plots and violin plots.
    
    Args:
        nested_measurements (dict): Nested measurements dictionary
        measurement_var (str): Variable being measured (e.g., 'weight', 'height')
        measurement_unit (str): Unit of measurement (e.g., 'kg', 'cm')
        real_discrepancies (array): Real discrepancy scores for each L1 unit
        measured_discrepancies (array): Measured discrepancy scores for each L1 unit
        real_ranks (array): Array of real ranks for each L1 unit
        measured_ranks (array): Array of measured ranks for each L1 unit
        figsize (tuple): Figure size (width, height)
    """
    # Get number of L1 units
    n_L1s = len(real_ranks)
    
    # Create figure
    fig, axs = plt.subplots(3, n_L1s, figsize=figsize, sharey='row', constrained_layout=True)
    
    # Font sizes
    TITLE_SIZE = 12
    LABEL_SIZE = 12
    TICK_SIZE = 10
    
    # Color scheme for L0s - create a distinct color for each L0 in a unit
    cmap = plt.get_cmap('Dark2')
    L0_colors = cmap(np.linspace(0, 1, max(len(nested_measurements[L1_id]) - 1 
                                                   for L1_id in nested_measurements 
                                                   if L1_id != 'metadata')))
    
    # Get order of L1 units based on real ranks
    L1_order = np.argsort(real_ranks)
    
    # Plot for each L1 unit
    for i, L1_idx in enumerate(L1_order):
        L1_id = f'L1_{L1_idx}'
        
        # Get measurement ranges for this unit to set axis limits
        all_measurements = []
        
        # Plot scatter plots for each L0
        for j, L0_id in enumerate(nested_measurements[L1_id]):
            if L0_id == 'L1_info':
                continue
                
            # Get measurements
            real_meas = nested_measurements[L1_id][L0_id]['real']['data'][measurement_var]
            L0_meas = nested_measurements[L1_id][L0_id]['L0']['data'][measurement_var]
            L1_meas = nested_measurements[L1_id][L0_id]['L1']['data'][measurement_var]
            
            all_measurements.extend(real_meas)
            all_measurements.extend(L0_meas)
            all_measurements.extend(L1_meas)
            
            # First row: Real vs L0
            axs[0, i].scatter(real_meas, L0_meas, color=L0_colors[j], alpha=0.5, 
                            label=f'L0_{j}', s=20)
            
            # Second row: L0 vs L1 (only for children measured by L1)
            L1_indices = nested_measurements[L1_id][L0_id]['L1']['data'].index
            axs[1, i].scatter(L1_meas, L0_meas.loc[L1_indices],
                            color=L0_colors[j], alpha=0.5, label=f'L0_{j}', s=20)
        
        # Set axis limits and add X=Y line for scatter plots
        axis_min = min(all_measurements)
        axis_max = max(all_measurements)
        line = np.linspace(axis_min, axis_max, 100)
        
        for row in [0, 1]:
            axs[row, i].plot(line, line, 'k--', alpha=0.5)
            axs[row, i].set_xlim(axis_min, axis_max)
            axs[row, i].set_ylim(axis_min, axis_max)
            if i == 0:
                if row == 0:
                    axs[row, i].set_ylabel(f'L0 {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
                else:
                    axs[row, i].set_ylabel(f'L0 {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
            axs[row, i].tick_params(axis='both', labelsize=TICK_SIZE)
        
        # Set x-labels
        axs[0, i].set_xlabel(f'Real {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
        axs[1, i].set_xlabel(f'L1 {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
        
        # Third row: Violin plots of discrepancies
        parts = axs[2, i].violinplot([real_discrepancies[L1_id], measured_discrepancies[L1_id]],
                                   positions=[1, 2],
                                   showmeans=True)
        
        # Color the violin plots
        colors = {'real': '#2c7bb6', 'measured': '#d7191c'}
        for pc, color in zip(parts['bodies'], colors.values()):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('black')
        
        # Set labels for violin plots
        axs[2, i].set_xticks([1, 2])
        axs[2, i].set_xticklabels(['Real', 'Measured'], fontsize=TICK_SIZE)
        axs[2, i].tick_params(axis='y', labelsize=TICK_SIZE)
        if i == 0:
            axs[2, i].set_ylabel(f'{measurement_var.capitalize()} discrepancy ({measurement_unit})', fontsize=LABEL_SIZE)
        
        # Add title showing ranks
        axs[0, i].set_title(f'Real rank: {real_ranks[L1_idx]}\nMeasured rank: {measured_ranks[L1_idx]}', 
                           fontsize=TITLE_SIZE)
    
    return fig

def L0_unit_classification_confidence(
    # Real parameters
    real_params,
    n_L1s,
    n_L0s_per_L1,
    n_children_per_L0,
    n_children_L1,
    n_children_L2,
    # WHO parameters
    haz_params,
    waz_params,
    whz_params_lying,
    whz_params_standing,
    # Analysis parameters
    measurement_var,
    n_L1_units_rewarded,
    # Distortion parameters for L0s
    real_percent_stunting=40,
    real_percent_underweight=40,
    real_percent_wasting=None,
    mean_percent_under_reporting_stunting=20,
    mean_percent_under_reporting_underweight=20,
    mean_percent_under_reporting_wasting=None,
    mean_bunch_factor_haz=0.1,
    mean_bunch_factor_waz=0.1,
    mean_bunch_factor_whz=0.1,
    sd_across_units_percent_under_reporting_stunting=5,
    sd_across_units_percent_under_reporting_underweight=5,
    sd_across_units_percent_under_reporting_wasting=5,
    sd_across_units_bunch_factor_haz=0.02,
    sd_across_units_bunch_factor_waz=0.02,
    sd_across_units_bunch_factor_whz=0.02,
    sd_within_units_percent_under_reporting_stunting=2,
    sd_within_units_percent_under_reporting_underweight=2,
    sd_within_units_percent_under_reporting_wasting=2,
    sd_within_units_bunch_factor_haz=0.01,
    sd_within_units_bunch_factor_waz=0.01,
    sd_within_units_bunch_factor_whz=0.01,
    # Distortion parameters for L1s
    mean_percent_copy=10,
    mean_collusion_index=0.2,
    sd_percent_copy=2,
    sd_collusion_index=0.1,
    error_mean_height_L1 = 0,
    error_sd_height_L1 = 1,
    error_mean_weight_L1 = 0,
    error_sd_weight_L1 = 0.1,
    bunch_factor_haz_L1 = 0.05,
    bunch_factor_waz_L1 = 0.05,
    bunch_factor_whz_L1 = 0.05,
    # Distortion parameters for L2
    error_mean_height_L2=0,
    error_sd_height_L2=1,
    error_mean_weight_L2=0,
    error_sd_weight_L2=0.1,
    drift_mean_height_L2=0,
    drift_sd_height_L2=0.1,
    drift_mean_weight_L2=0,
    drift_sd_weight_L2=0.05,

    random_seed=None,
    n_simulations=100,  # Add n_simulations parameter
):
    """
    Analyze classification confidence for different parameter combinations.
    
    Args:
        [Original args from generate_nested_measurements except L0_params_list, L1_params_list]
        [All args from generate_nested_distortion_parameters]
        measurement_var (str): Variable to analyze (e.g., 'weight', 'height')
        measurement_unit (str): Unit of measurement (e.g., 'kg', 'cm')
        n_L1_units_rewarded (int): Number of top L1 units to consider
        n_simulations (int): Number of simulations to run for each parameter combination
        make_plots (bool): Whether to show diagnostic plots
    
    Returns:
        tuple: (DataFrame with results, plots)
    """
    # Convert inputs to lists if they're not already
    n_L0s_list = [n_L0s_per_L1] if isinstance(n_L0s_per_L1, int) else n_L0s_per_L1
    n_children_L0_list = [n_children_per_L0] if isinstance(n_children_per_L0, int) else n_children_per_L0
    n_children_L1_list = [n_children_L1] if isinstance(n_children_L1, int) else n_children_L1
    
    # Check if n_L1_units_rewarded is valid
    if n_L1_units_rewarded >= n_L1s:
        raise ValueError(f"n_L1_units_rewarded ({n_L1_units_rewarded}) must be less than n_L1s ({n_L1s})")
    
    # Create empty lists to store results
    results = []
    
    # Iterate over all parameter combinations
    for n_L0s in n_L0s_list:
        for n_children_L0 in n_children_L0_list:
            for n_children_L1 in n_children_L1_list:
                sim_real_ranks = []
                sim_overlaps = []
                warning_count = 0  # Track warnings for this parameter combination

                
                # Run multiple simulations
                for sim in range(n_simulations):
                    # Generate distortion parameters
                    L0_params_list, L1_params_list, L2_params_dict = generate_nested_distortion_parameters(
                        n_L1s=n_L1s,
                        n_L0s_per_L1=n_L0s,
                        # Real percentages
                        real_percent_stunting=real_percent_stunting,
                        real_percent_underweight=real_percent_underweight,
                        real_percent_wasting=real_percent_wasting,
                        # L0 parameter means
                        mean_percent_under_reporting_stunting=mean_percent_under_reporting_stunting,
                        mean_percent_under_reporting_underweight=mean_percent_under_reporting_underweight,
                        mean_percent_under_reporting_wasting=mean_percent_under_reporting_wasting,
                        mean_bunch_factor_haz=mean_bunch_factor_haz,
                        mean_bunch_factor_waz=mean_bunch_factor_waz,
                        mean_bunch_factor_whz=mean_bunch_factor_whz,
                        # L0 parameter standard deviations across units
                        sd_across_units_percent_under_reporting_stunting=sd_across_units_percent_under_reporting_stunting,
                        sd_across_units_percent_under_reporting_underweight=sd_across_units_percent_under_reporting_underweight,
                        sd_across_units_percent_under_reporting_wasting=sd_across_units_percent_under_reporting_wasting,
                        sd_across_units_bunch_factor_haz=sd_across_units_bunch_factor_haz,
                        sd_across_units_bunch_factor_waz=sd_across_units_bunch_factor_waz,
                        sd_across_units_bunch_factor_whz=sd_across_units_bunch_factor_whz,
                        # L0 parameter standard deviations within units
                        sd_within_units_percent_under_reporting_stunting=sd_within_units_percent_under_reporting_stunting,
                        sd_within_units_percent_under_reporting_underweight=sd_within_units_percent_under_reporting_underweight,
                        sd_within_units_percent_under_reporting_wasting=sd_within_units_percent_under_reporting_wasting,
                        sd_within_units_bunch_factor_haz=sd_within_units_bunch_factor_haz,
                        sd_within_units_bunch_factor_waz=sd_within_units_bunch_factor_waz,
                        sd_within_units_bunch_factor_whz=sd_within_units_bunch_factor_whz,
                        # L1 parameters
                        mean_percent_copy=mean_percent_copy,
                        mean_collusion_index=mean_collusion_index,
                        sd_percent_copy=sd_percent_copy,
                        sd_collusion_index=sd_collusion_index,
                        error_mean_height_L1 =error_mean_height_L1,
                        error_sd_height_L1 = error_sd_height_L1,
                        error_mean_weight_L1 = error_mean_weight_L1,
                        error_sd_weight_L1 = error_sd_weight_L1,
                        bunch_factor_haz_L1 = bunch_factor_haz_L1,
                        bunch_factor_waz_L1 = bunch_factor_waz_L1,
                        bunch_factor_whz_L1 = bunch_factor_whz_L1,
                        # L2 parameters
                        error_mean_height_L2=error_mean_height_L2,
                        error_sd_height_L2=error_sd_height_L2,
                        error_mean_weight_L2=error_mean_weight_L2,
                        error_sd_weight_L2=error_sd_weight_L2,
                        drift_mean_height_L2=drift_mean_height_L2,
                        drift_sd_height_L2=drift_sd_height_L2,
                        drift_mean_weight_L2=drift_mean_weight_L2,
                        drift_sd_weight_L2=drift_sd_weight_L2,
                        random_seed=random_seed+sim if random_seed else None
                    )
                    
                    # Generate nested measurements
                    nested_measurements = generate_nested_measurements(
                        real_params=real_params,
                        L0_params_list=L0_params_list,
                        L1_params_list=L1_params_list,
                        L2_params_dict=L2_params_dict,
                        n_L1s=n_L1s,
                        n_L0s_per_L1=n_L0s,
                        n_children_per_L0=n_children_L0,
                        n_children_L1=n_children_L1,
                        n_children_L2=n_children_L2,
                        haz_params=haz_params,
                        waz_params=waz_params,
                        whz_params_lying=whz_params_lying,
                        whz_params_standing=whz_params_standing,
                        make_plots=False
                    )
                    
                    # Check for warnings in the measurements
                    warning_found = False
                    for L1_id in nested_measurements:
                        if L1_id == 'metadata':
                            continue

                        # Check each L0 in this L1 unit
                        for L0_id in nested_measurements[L1_id]:
                            if L0_id == 'L1_info':
                                continue

                            # Check the L0 data first
                            if 'bunching_warning_haz' in nested_measurements[L1_id][L0_id]['L0']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L0']['metadata']['bunching_warning_haz']:
                                    warning_found = True
                                    break
                            if 'bunching_warning_waz' in nested_measurements[L1_id][L0_id]['L0']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L0']['metadata']['bunching_warning_waz']:
                                    warning_found = True
                                    break
                            if 'bunching_warning_whz' in nested_measurements[L1_id][L0_id]['L0']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L0']['metadata']['bunching_warning_whz']:
                                    warning_found = True
                                    break
                            # Check the L1 data too
                            if 'bunching_warning_haz' in nested_measurements[L1_id][L0_id]['L1']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L1']['metadata']['bunching_warning_haz']:
                                    warning_found = True
                                    break
                            if 'bunching_warning_waz' in nested_measurements[L1_id][L0_id]['L1']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L1']['metadata']['bunching_warning_waz']:
                                    warning_found = True
                                    break
                            if 'bunching_warning_whz' in nested_measurements[L1_id][L0_id]['L1']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L1']['metadata']['bunching_warning_whz']:
                                    warning_found = True
                                    break
                        if warning_found:
                            break
                    if warning_found:
                        warning_count += 1

                    # Calculate ranks
                    real_ranks, measured_ranks, _, _ = calculate_ranks_L0_units(
                        nested_measurements, 
                        measurement_var
                    )
                    
                    # Find real rank of L1 unit with best measured rank
                    best_measured_idx = np.argmin(measured_ranks)
                    sim_real_ranks.append(real_ranks[best_measured_idx])
                    
                    # Find overlap between top units
                    top_real = np.where(real_ranks <= n_L1_units_rewarded)[0]
                    top_measured = np.where(measured_ranks <= n_L1_units_rewarded)[0]
                    sim_overlaps.append(len(set(top_real) & set(top_measured)))
                
                # Print warning count for this parameter combination
                if warning_count > 0:
                    print(f"\nWarning: For parameter combination:")
                    print(f"n_L0s_per_L1={n_L0s}, n_children_per_L0={n_children_L0}, n_children_L1={n_children_L1}")
                    print(f"Percent shift was too high in {warning_count} out of {n_simulations} simulations")
                

                # Store results with mean and SEM across simulations
                results.append({
                    'n_L0s_per_L1': n_L0s,
                    'n_children_per_L0': n_children_L0,
                    'n_children_L1': n_children_L1,
                    'real_rank_mean': np.mean(sim_real_ranks),
                    'real_rank_sem': np.std(sim_real_ranks) / np.sqrt(n_simulations),
                    'n_overlap_mean': np.mean(sim_overlaps),
                    'n_overlap_sem': np.std(sim_overlaps) / np.sqrt(n_simulations)
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def plot_L0_unit_classification_confidence_vs_parameters(
    n_L0s_list,
    n_children_L0_list,
    n_children_L1_list,
    results_df,
    n_L1_units_rewarded
):
    
    """
    Plot classification confidence results from L0_unit_classification_confidence function.
    
    Args:
        results_df (pd.DataFrame): DataFrame with results from L0_unit_classification_confidence

    Returns:
        fig_list (list): List of matplotlib Figure objects

    """

    # Create plots
    FONT_SIZE = 14
    fig_list = []
    
    # Determine which parameters are varying
    plot_vars = []
    if len(n_L0s_list) > 1:
        plot_vars.append('n_L0s_per_L1')
    if len(n_children_L0_list) > 1:
        plot_vars.append('n_children_per_L0')
    if len(n_children_L1_list) > 1:
        plot_vars.append('n_children_L1')
    
    # Colors for different parameter values
    color_map = plt.cm.get_cmap('Dark2')
    
    for var in plot_vars:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Find other varying parameters
        other_vars = [v for v in plot_vars if v != var]
        
        if other_vars:
            # Plot lines for each value of the other parameter
            other_var = other_vars[0]
            other_values = sorted(results_df[other_var].unique())
            colors = color_map(np.linspace(0, 1, len(other_values)))
            
            for i, val in enumerate(other_values):
                mask = results_df[other_var] == val
                x = results_df[mask][var]
                
                # Plot real rank
                y = results_df[mask]['real_rank_mean']
                yerr = results_df[mask]['real_rank_sem']
                ax1.errorbar(x, y, yerr=yerr, fmt='o-', color=colors[i], 
                           label=f'{other_var}={val}', linewidth=2, markersize=8)
                
                # Plot overlap
                y = results_df[mask]['n_overlap_mean']
                yerr = results_df[mask]['n_overlap_sem']
                ax2.errorbar(x, y, yerr=yerr, fmt='o-', color=colors[i], 
                           label=f'{other_var}={val}', linewidth=2, markersize=8)
        else:
            # Single line if no other parameters vary
            x = results_df[var]
            
            # Plot real rank
            y = results_df['real_rank_mean']
            yerr = results_df['real_rank_sem']
            ax1.errorbar(x, y, yerr=yerr, fmt='ko-', linewidth=2, markersize=8)
            
            # Plot overlap
            y = results_df['n_overlap_mean']
            yerr = results_df['n_overlap_sem']
            ax2.errorbar(x, y, yerr=yerr, fmt='ko-', linewidth=2, markersize=8)
        
        # Set labels and formatting
        ax1.set_xlabel(var.replace('_', ' ').title(), fontsize=FONT_SIZE)
        ax1.set_ylabel('Real Rank of Best\nMeasured Unit', fontsize=FONT_SIZE)
        ax1.tick_params(axis='both', labelsize=FONT_SIZE-2)
        ax1.grid(True)
        if other_vars:
            ax1.legend(fontsize=FONT_SIZE-2)
        
        ax2.set_xlabel(var.replace('_', ' ').title(), fontsize=FONT_SIZE)
        ax2.set_ylabel(f'Number of True Top {n_L1_units_rewarded}\nUnits Identified', fontsize=FONT_SIZE)
        ax2.tick_params(axis='both', labelsize=FONT_SIZE-2)
        ax2.grid(True)
        if other_vars:
            ax2.legend(fontsize=FONT_SIZE-2)
        
        plt.tight_layout()
        fig_list.append(fig)

    return fig_list

def L0_classification_confidence_vs_L2_L1_discrepancy(
    # Real parameters
    real_params,
    n_L1s,
    n_L0s_per_L1,
    n_children_per_L0,
    n_children_L1,
    n_children_L2,
    # WHO parameters
    haz_params,
    waz_params,
    whz_params_lying,
    whz_params_standing,
    # Analysis parameters
    measurement_var,
    n_L0s_rewarded_per_L1,
    discrepancy_method='simple_difference',
    # Distortion parameters for L0s
    real_percent_stunting=40,
    real_percent_underweight=40,
    real_percent_wasting=None,
    mean_percent_under_reporting_stunting=20,
    mean_percent_under_reporting_underweight=20,
    mean_percent_under_reporting_wasting=None,
    mean_bunch_factor_haz=0.1,
    mean_bunch_factor_waz=0.1,
    mean_bunch_factor_whz=0.1,
    error_mean_height_all_L0s = 0,
    error_sd_height_all_L0s = 1,
    error_mean_weight_all_L0s = 0,
    error_sd_weight_all_L0s = 0.1,
    sd_across_units_percent_under_reporting_stunting=5,
    sd_across_units_percent_under_reporting_underweight=5,
    sd_across_units_percent_under_reporting_wasting=5,
    sd_across_units_bunch_factor_haz=0.02,
    sd_across_units_bunch_factor_waz=0.02,
    sd_across_units_bunch_factor_whz=0.02,
    sd_within_units_percent_under_reporting_stunting=2,
    sd_within_units_percent_under_reporting_underweight=2,
    sd_within_units_percent_under_reporting_wasting=2,
    sd_within_units_bunch_factor_haz=0.01,
    sd_within_units_bunch_factor_waz=0.01,
    sd_within_units_bunch_factor_whz=0.01,
    # Distortion parameters for L1s
    mean_percent_copy=10,
    mean_collusion_index=0.2,
    sd_percent_copy=2,
    sd_collusion_index=0.1,
    error_mean_height_L1=0,
    error_sd_height_L1=1,
    error_mean_weight_L1=0,
    error_sd_weight_L1=0.1,
    bunch_factor_haz_L1=0.05,
    bunch_factor_waz_L1=0.05,
    bunch_factor_whz_L1=0.05,
    # Distortion parameters for L2
    error_mean_height_L2=0,
    error_sd_height_L2=1,
    error_mean_weight_L2=0,
    error_sd_weight_L2=0.1,
    drift_mean_height_L2=0,
    drift_sd_height_L2=0.1,
    drift_mean_weight_L2=0,
    drift_sd_weight_L2=0.05,
    random_seed=None,
    n_simulations=100,
):
    """
    Analyze L0 classification confidence versus L2-L1 discrepancy scores.
    
    Args:
        [Same as L0_unit_classification_confidence except n_L1_units_rewarded replaced with n_L0s_rewarded_per_L1]
        n_L0s_rewarded_per_L1 (int): Number of top L0s to reward within each L1 unit
        discrepancy_method (str): Method to calculate discrepancy scores
        
    Returns:
        tuple: (n_real_L0s_rewarded, L2_L1_discrepancies, fig)
    """
    # Validation
    if n_L0s_rewarded_per_L1 > n_L0s_per_L1:
        raise ValueError(f"n_L0s_rewarded_per_L1 ({n_L0s_rewarded_per_L1}) must be <= n_L0s_per_L1 ({n_L0s_per_L1})")
    
    # Initialize lists
    n_real_L0s_rewarded = []
    L2_L1_discrepancies = []
    params = {
        'percent_copy': [],
        'collusion_index': []
    }
    warning_count = 0
    
    # Run simulations
    for sim in range(n_simulations):

        # Generate distortion parameters
        L0_params_list, L1_params_list, L2_params_dict = generate_nested_distortion_parameters(
            n_L1s=n_L1s,
            n_L0s_per_L1=n_L0s_per_L1,
            # Real percentages
            real_percent_stunting=real_percent_stunting,
            real_percent_underweight=real_percent_underweight,
            real_percent_wasting=real_percent_wasting,
            # L0 parameter means
            mean_percent_under_reporting_stunting=mean_percent_under_reporting_stunting,
            mean_percent_under_reporting_underweight=mean_percent_under_reporting_underweight,
            mean_percent_under_reporting_wasting=mean_percent_under_reporting_wasting,
            mean_bunch_factor_haz=mean_bunch_factor_haz,
            mean_bunch_factor_waz=mean_bunch_factor_waz,
            mean_bunch_factor_whz=mean_bunch_factor_whz,
            error_mean_height_all_L0s = error_mean_height_all_L0s,
            error_sd_height_all_L0s = error_sd_height_all_L0s,
            error_mean_weight_all_L0s = error_mean_weight_all_L0s,
            error_sd_weight_all_L0s = error_sd_weight_all_L0s,
            # L0 parameter standard deviations across units
            sd_across_units_percent_under_reporting_stunting=sd_across_units_percent_under_reporting_stunting,
            sd_across_units_percent_under_reporting_underweight=sd_across_units_percent_under_reporting_underweight,
            sd_across_units_percent_under_reporting_wasting=sd_across_units_percent_under_reporting_wasting,
            sd_across_units_bunch_factor_haz=sd_across_units_bunch_factor_haz,
            sd_across_units_bunch_factor_waz=sd_across_units_bunch_factor_waz,
            sd_across_units_bunch_factor_whz=sd_across_units_bunch_factor_whz,
            # L0 parameter standard deviations within units
            sd_within_units_percent_under_reporting_stunting=sd_within_units_percent_under_reporting_stunting,
            sd_within_units_percent_under_reporting_underweight=sd_within_units_percent_under_reporting_underweight,
            sd_within_units_percent_under_reporting_wasting=sd_within_units_percent_under_reporting_wasting,
            sd_within_units_bunch_factor_haz=sd_within_units_bunch_factor_haz,
            sd_within_units_bunch_factor_waz=sd_within_units_bunch_factor_waz,
            sd_within_units_bunch_factor_whz=sd_within_units_bunch_factor_whz,
            # L1 parameters
            mean_percent_copy=mean_percent_copy,
            mean_collusion_index=mean_collusion_index,
            sd_percent_copy=sd_percent_copy,
            sd_collusion_index=sd_collusion_index,
            error_mean_height_L1=error_mean_height_L1,
            error_sd_height_L1=error_sd_height_L1,
            error_mean_weight_L1=error_mean_weight_L1,
            error_sd_weight_L1=error_sd_weight_L1,
            bunch_factor_haz_L1=bunch_factor_haz_L1,
            bunch_factor_waz_L1=bunch_factor_waz_L1,
            bunch_factor_whz_L1=bunch_factor_whz_L1,
            # L2 parameters
            error_mean_height_L2=error_mean_height_L2,
            error_sd_height_L2=error_sd_height_L2,
            error_mean_weight_L2=error_mean_weight_L2,
            error_sd_weight_L2=error_sd_weight_L2,
            drift_mean_height_L2=drift_mean_height_L2,
            drift_sd_height_L2=drift_sd_height_L2,
            drift_mean_weight_L2=drift_mean_weight_L2,
            drift_sd_weight_L2=drift_sd_weight_L2,
            random_seed=random_seed
        )
    
        # Generate nested measurements
        nested_measurements = generate_nested_measurements(
            real_params=real_params,
            L0_params_list=L0_params_list,
            L1_params_list=L1_params_list,
            L2_params_dict=L2_params_dict,
            n_L1s=n_L1s,
            n_L0s_per_L1=n_L0s_per_L1,
            n_children_per_L0=n_children_per_L0,
            n_children_L1=n_children_L1,
            n_children_L2=n_children_L2,
            haz_params=haz_params,
            waz_params=waz_params,
            whz_params_lying=whz_params_lying,
            whz_params_standing=whz_params_standing,
            make_plots=False
        )
        L1_L2_pairwise_data = get_L1_L2_pairwise_data(nested_measurements)

        # Check for warnings
        warning_found = False
        for L1_id in nested_measurements:
            if L1_id == 'metadata':
                continue
            
            for L0_id in nested_measurements[L1_id]:
                if L0_id == 'L1_info':
                    continue
                
                # Check L0 warnings
                for measure in ['haz', 'waz', 'whz']:
                    warning_key = f'bunching_warning_{measure}'
                    if nested_measurements[L1_id][L0_id]['L0']['metadata'].get(warning_key, False):
                        warning_found = True
                        break
                    if nested_measurements[L1_id][L0_id]['L1']['metadata'].get(warning_key, False):
                        warning_found = True
                        break
                if warning_found:
                    break
            if warning_found:
                break
        
        if warning_found:
            warning_count += 1
        
        # Calculate L0 ranks within each L1 unit
        L0_ranks = calculate_ranks_L0s(nested_measurements, measurement_var, method=discrepancy_method)
        
        # Calculate overlap and L2-L1 discrepancy for each L1 unit
        L1_no = 0
        for L1_id in L0_ranks:

            real_ranks = L0_ranks[L1_id]['real_ranks']
            measured_ranks = L0_ranks[L1_id]['measured_ranks']
            
            # Find overlap in top n_L0s_rewarded_per_L1
            top_real = np.where(real_ranks <= n_L0s_rewarded_per_L1)[0]
            top_measured = np.where(measured_ranks <= n_L0s_rewarded_per_L1)[0]
            overlap = len(set(top_real) & set(top_measured))
            n_real_L0s_rewarded.append(overlap)
            
            # Calculate L2-L1 discrepancy for this L1 unit            
            
            disc = calculate_discrepancy_scores(
                L1_L2_pairwise_data[L1_id]['L1'],
                L1_L2_pairwise_data[L1_id]['L2'],
                measurement_var,
                discrepancy_method,
                make_plot=False
            )
            L2_L1_discrepancies.append(abs(disc.mean()))

            # Store L1 parameters for this L1 unit
            params['percent_copy'].append(L1_params_list[L1_no]['percent_copy'])
            params['collusion_index'].append(L1_params_list[L1_no]['collusion_index'])

            L1_no += 1
    
    # Print warning statistics
    if warning_count > 0:
        print(f"\nWarning: Percent shift was too high in {warning_count} out of {n_simulations} simulations")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(L2_L1_discrepancies, n_real_L0s_rewarded, alpha=0.5, s=50, color='steelblue')
    ax.set_xlabel(f'L2-L1 Discrepancy ({measurement_var})', fontsize=14)
    ax.set_ylabel(f'Number of True Top {n_L0s_rewarded_per_L1} L0s Identified', fontsize=14)
    ax.set_title('L0 Classification Confidence vs L2-L1 Discrepancy', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    
    return n_real_L0s_rewarded, L2_L1_discrepancies, params, fig