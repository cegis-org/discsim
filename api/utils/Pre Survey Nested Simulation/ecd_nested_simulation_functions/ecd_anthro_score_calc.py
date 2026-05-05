import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_haz(height, age, sex, haz_params):
    """
    Calculate Height-for-Age Z-scores (HAZ) using the WHO growth standards.

    Args:
        height (pd.Series): Height measurements.
        age (pd.Series): Age measurements.
        sex (pd.Series): Sex (gender) of the children.
        loh (pd.Series): Whether height was measured standing or lying down.
        haz_params (pd.DataFrame): HAZ parameters from WHO growth standards.

    Returns:
        pd.Series: HAZ scores.
    """
    # Make sure height, age, sex and loh are the same length, else throw an error
    if not (len(height) == len(age) == len(sex)):
        raise ValueError("Input series must have the same length")

    # Get M, S, L values for the age and sex
    m, s, l = get_msl(age, sex, haz_params)
    haz = get_anthro_zscore(height.dropna(), m, s, l)
    return haz

def height_from_haz(haz, age, sex, haz_params):
    """
    Calculate height from Height-for-Age Z-scores (HAZ) using the WHO growth standards.

    Args:
        haz (pd.Series): HAZ scores.
        age (pd.Series): Age measurements.
        sex (pd.Series): Sex (gender) of the children.
        loh (pd.Series): Whether height was measured standing or lying down.
        haz_params (pd.DataFrame): HAZ parameters from WHO growth standards.

    Returns:
        pd.Series: Height measurements.
    """
    # Make sure haz, age, sex and loh are the same length, else throw an error
    if not (len(haz) == len(age) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # Get M, S, L values for the age and sex
    m, s, l = get_msl(age, sex, haz_params)
    height = invert_anthro_zscore(haz, m, s, l)
    return height

def calculate_waz(weight, age, sex, waz_params):
    """
    Calculate Weight-for-Age Z-scores (WAZ) using the WHO growth standards.

    Args:
        weight (pd.Series): Weight measurements.
        age (pd.Series): Age measurements.
        sex (pd.Series): Sex (gender) of the children.
        waz_params (pd.DataFrame): WAZ parameters from WHO growth standards.

    Returns:
        pd.Series: WAZ scores.
    """
    # Make sure weight, age, and sex are the same length, else throw an error
    if not (len(weight) == len(age) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # Get M, S, L values for the age and sex
    m, s, l = get_msl(age, sex, waz_params)
    waz = get_anthro_zscore(weight, m, s, l)
    return waz

def weight_from_waz(waz, age, sex, waz_params):
    """
    Calculate weight from Weight-for-Age Z-scores (WAZ) using the WHO growth standards.

    Args:
        waz (pd.Series): WAZ scores.
        age (pd.Series): Age measurements.
        sex (pd.Series): Sex (gender) of the children.
        waz_params (pd.DataFrame): WAZ parameters from WHO growth standards.

    Returns:
        pd.Series: Weight measurements.
    """
    # Make sure waz, age, and sex are the same length, else throw an error
    if not (len(waz) == len(age) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # Get M, S, L values for the age and sex
    m, s, l = get_msl(age, sex, waz_params)
    weight = invert_anthro_zscore(waz, m, s, l)
    return weight

def weight_from_whz(whz, height, sex, loh, whz_params_lying, whz_params_standing):
    """
    Calculate weight from Weight-for-Height Z-scores (WHZ) using the WHO growth standards.

    Args:
        whz (pd.Series): WHZ scores.
        height (pd.Series): Height measurements.
        sex (pd.Series): Sex (gender) of the children.
        whz_params (pd.DataFrame): WHZ parameters from WHO growth standards.

    Returns:
        pd.Series: Weight measurements.
    """
    # Make sure whz, height, and sex are the same length, else throw an error
    if not (len(whz) == len(height) == len(sex) == len(loh)):
        raise ValueError("Input series must have the same length")
    
    # Split height, whz and sex into two groups based on loh
    height_0 = height[loh == 1]  # Lying height
    whz_0 = whz[loh == 1]  # WHZ corresponding to lying height
    sex_0 = sex[loh == 1]  # Sex corresponding to lying height

    height_1 = height[loh == 2]  # Standing height
    whz_1 = whz[loh == 2]  # WHZ corresponding to standing height
    sex_1 = sex[loh == 2]  # Sex corresponding to standing height

    # Get M, S, L values for the height and sex for lying height
    # Create a variable called height_adjusted in which each value from height_0 is replaced with the closest value from whz_params_lying['__000002']
    height_array = np.array(whz_params_lying['__000002'])
    height_adjusted = pd.Series(height_array[np.abs(height_array[:, None] - height_0.values).argmin(axis=0)]) 
    m, s, l = get_msl(pd.Series(height_adjusted), sex_0, whz_params_lying, age_label = '__000002')
    weight_0 = invert_anthro_zscore(whz_0, m, s, l)

    # Get M, S, L values for the height and sex for standing height
    # Create a variable called height_adjusted in which each value from height_1 is replaced with the closest value from whz_params_standing['__000003']
    height_array = np.array(whz_params_standing['__000003'])
    height_adjusted = pd.Series(height_array[np.abs(height_array[:, None] - height_1.values).argmin(axis=0)]) 
    m, s, l = get_msl(pd.Series(height_adjusted), sex_1, whz_params_standing, age_label = '__000003')
    weight_1 = invert_anthro_zscore(whz_1, m, s, l)

    # Combine the results
    weight = pd.concat([weight_0, weight_1]).sort_index()
    return weight

def calculate_whz(height, weight, sex, loh, whz_params_lying, whz_params_standing):
    """
    Calculate Weight-for-Height Z-scores (WHZ) using the WHO growth standards.

    Args:
        height (pd.Series): Height measurements.
        weight (pd.Series): Weight measurements.
        age (pd.Series): Age measurements.
        loh (pd.Series): Whether height was measured standing or lying down.
        whz_params_lying (pd.DataFrame): WHZ parameters from WHO growth standards for heights measured lying down.
        whz_params_standing (pd.DataFrame): WHZ parameters from WHO growth standards for heights measured standing up.

    Returns:
        pd.Series: WHZ scores.
    """
    # Make sure height, weight, age, loh and sex are the same length, else throw an error
    if not (len(height) == len(weight) == len(loh) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # Split height, weight and sex into two groups based on loh
    height_0 = height[loh == 1]  # Lying height
    weight_0 = weight[loh == 1] # Weight corresponding to lying height
    sex_0 = sex[loh == 1] # Sex corresponding to lying height

    height_1 = height[loh == 2]  # Standing height
    weight_1 = weight[loh == 2] # Weight corresponding to standing height
    sex_1 = sex[loh == 2] # Sex corresponding to standing height

    # Get M, S, L values for the age and sex for lying height
    # Create a variable called height_adjusted in which each value from height_0 is replaced with the closest value from whz_params_lying['__000002']
    height_array = np.array(whz_params_lying['__000002'])
    height_adjusted = pd.Series(height_array[np.abs(height_array[:, None] - height_0.values).argmin(axis=0)]) 
    m, s, l = get_msl(pd.Series(height_adjusted), sex_0, whz_params_lying, age_label = '__000002')
    whz_0 = get_anthro_zscore(weight_0.dropna(), m, s, l)

    # Get M, S, L values for the age and sex for standing height
    # Create a variable called height_adjusted in which each value from height_1 is replaced with the closest value from whz_params_standing['__000003']
    height_array = np.array(whz_params_standing['__000003'])
    height_adjusted = pd.Series(height_array[np.abs(height_array[:, None] - height_1.values).argmin(axis=0)]) 
    m, s, l = get_msl(pd.Series(height_adjusted), sex_1, whz_params_standing, age_label = '__000003')
    whz_1 = get_anthro_zscore(weight_1.dropna(), m, s, l)
    
    # Return combined whz values
    whz = pd.concat([whz_0, whz_1]).sort_index()
    return whz

def get_anthro_zscore(y, m, s, l):
    """
    Calculate the anthropometric Z-score using the WHO growth standards.

    Args:
        y (pd.Series): The measurement (e.g., height, weight).
        m (pd.Series): The median value for the reference population.
        s (pd.Series): The standard deviation for the reference population.
        l (pd.Series): The lambda (Box-Cox) transformation parameter.

    Returns:
        pd.Series: The Z-scores.
    """
    z = (np.power(y/m, l) - 1 )/ (s * l)
    return z

def invert_anthro_zscore(z, m, s, l):
    """
    Invert the anthropometric Z-score to obtain the original measurement.

    Args:
        z (pd.Series): The Z-scores.
        m (pd.Series): The median value for the reference population.
        s (pd.Series): The standard deviation for the reference population.
        l (pd.Series): The lambda (Box-Cox) transformation parameter.

    Returns:
        pd.Series: The original measurements.
    """
    y = m * np.power((1 + z * s * l), 1/l)
    return y

def get_msl(age, sex, params, age_label = '_agedays', sex_label = '__000001', verbose=False):

    """
    Get M, S, L values for a specific age and sex from the WHO growth standards.

    Args:
        params (pd.DataFrame): The growth standards parameters.
        age_label (int): The age in days.
        sex_label (int): The sex (1 for male, 2 for female).

    Returns:
        tuple: A tuple containing the M, S, L values.
    """
    n_vals = len(age)
    m, s, l = (np.empty(n_vals), np.empty(n_vals), np.empty(n_vals))

    # Get unique ages from params for mapping
    unique_ages = np.array(params[age_label])

    # Split params by sex
    split_idx = len(unique_ages) // 2

    # Verify that  first half is sex==1, second half is sex==2
    if not (np.all(params[sex_label][:split_idx] == 1) and np.all(params[sex_label][split_idx:] == 2)):
        raise ValueError("Inconsistent sex labels in params")

    # Create lookup dicts for each sex
    lookup_m = {
        1: dict(zip(params[age_label][:split_idx], params['m'][:split_idx])),
        2: dict(zip(params[age_label][split_idx:], params['m'][split_idx:]))
    }
    lookup_s = {
        1: dict(zip(params[age_label][:split_idx], params['s'][:split_idx])),
        2: dict(zip(params[age_label][split_idx:], params['s'][split_idx:]))
    }
    lookup_l = {
        1: dict(zip(params[age_label][:split_idx], params['l'][:split_idx])),
        2: dict(zip(params[age_label][split_idx:], params['l'][split_idx:]))
    }

    if verbose:
        for idx, i in tqdm(enumerate(age.dropna().index)):
            sex_i = np.array(sex)[idx]
            age_i = np.array(age)[idx]
            m[idx] = lookup_m[sex_i].get(age_i, np.nan)
            s[idx] = lookup_s[sex_i].get(age_i, np.nan)
            l[idx] = lookup_l[sex_i].get(age_i, np.nan)
    else:
        for idx, i in enumerate(age.dropna().index):
            sex_i = np.array(sex)[idx]
            age_i = np.array(age)[idx]
            m[idx] = lookup_m[sex_i].get(age_i, np.nan)
            s[idx] = lookup_s[sex_i].get(age_i, np.nan)
            l[idx] = lookup_l[sex_i].get(age_i, np.nan)

    return m, s, l
