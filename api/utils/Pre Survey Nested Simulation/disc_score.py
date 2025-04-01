import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def discrepancy_score(subordinate_variable, supervisor_variable, method):
    """Calculate the discrepancy score between two variables.
    The discrepancy score is a measure of the difference between the two variables.
    
    Inputs:
    subordinate_variable (array, float except if method is 'percent_non_match'): values measured by subordinate
    supervisor_variable (array): must be of same length and datatype as subordinate_variable, values measured by supervisor
    method (string): One of the following options:
        1. 'percent_difference': discrepancy score is the signed difference between subordinate and supervisor measurements, as a percentage of the supervisor measurement
        2. 'absolute_difference': discrepancy score is the absolute difference between subordiante and supervisor measurements
        3. 'absolute_percent_difference': discrepancy score is the absolute difference between subordinate and supervisor measurements, as a percentage of the supervisor measurement
        4. 'simple_difference': discrepancy score is the signed difference between subordinate and supervisor measurements
        5. 'percent_non_match': discrepancy score is the percentage of measurements that are not exactly same between subordinate and supervisor. If method is 'percent_non_match', 
                                subordinate_variable and supervisor_variable may be of any datatype.
                                
    Outputs:
    discrepancy score (float): Average of discrepancy score for all measurements (except if method is percent_non_match, in which case it is simply the 
                               percentage of measurements that are not exactly same between subordinate and supervisor).
    
    """
    
    # Flatten variable arrays and make sure they are numpy arrays
    subordinate_variable = np.reshape(subordinate_variable, [-1])
    supervisor_variable = np.reshape(supervisor_variable, [-1])
    
    # Step 1: check that the two variables are the same length
    if len(subordinate_variable) != len(supervisor_variable):
        raise ValueError("The two variables must be the same length.")
    
    # Step 2: check that the two variables are the same type
    if type(subordinate_variable[0]) != type(supervisor_variable[0]):
        raise TypeError("The two variables must be the same type.")
    
    # Step 3: calculate the discrepancy score
    if method == "percent_difference":
        discrepancy_score = np.mean(np.divide((subordinate_variable - supervisor_variable), supervisor_variable)) * 100
    elif method == "absolute_difference":
        discrepancy_score = np.mean(abs(subordinate_variable - supervisor_variable))
    elif method == "absolute_percent_difference":
        discrepancy_score = np.mean(abs((subordinate_variable - supervisor_variable) / supervisor_variable * 100))
    elif method == "simple_difference":
        discrepancy_score = np.mean(subordinate_variable - supervisor_variable)
    elif method == "percent_non_match":
        discrepancy_score = np.sum(subordinate_variable != supervisor_variable) / len(subordinate_variable) * 100
    elif method == 'directional_percent_non_match':
        # If the subordinate variable is 0 and the supervisor variable is 1, or the subordinate variable is False
        # and the supervisor variable is True, return 0, else return the percentage of non-matches
        discrepancy_score = np.sum(np.where((subordinate_variable == 0) & (supervisor_variable == 1), 0, 
                  (subordinate_variable != supervisor_variable).astype(int)))/len(subordinate_variable) * 100
    elif method == "directional_difference":
        # If the subordinate variable is less than the supervisor variable, return 0, else return the difference
        discrepancy_score = np.mean(np.where(subordinate_variable < supervisor_variable, 0, 
                                             subordinate_variable - supervisor_variable))
    else:
        raise ValueError("The method must be one of the following: percent_difference, absolute_difference, "
        "absolute_percent_difference, simple_difference, percent_non_match, or directional_difference.")
    
    return discrepancy_score


def bootstrap_distribution(subordinate_variable, supervisor_variable, method, n_iterations = 100000, ax = None):
    """Generate a distribution of discrepancy scores between the two variables using bootstrapping.
    
    Inputs:
    subordinate_variable (array, float except if method is 'percent_non_match'): values measured by subordinate
    supervisor_variable (array): must be of same length and datatype as subordinate_variable, values measured by supervisor
    method (string): One of the following options:
        1. 'percent_difference': discrepancy score is the signed difference between subordinate and supervisor measurements, as a percentage of the supervisor measurement
        2. 'absolute_difference': discrepancy score is the absolute difference between subordiante and supervisor measurements
        3. 'absolute_percent_difference': discrepancy score is the absolute difference between subordinate and supervisor measurements, as a percentage of the supervisor measurement
        4. 'simple_difference': discrepancy score is the signed difference between subordinate and supervisor measurements
        5. 'percent_non_match': discrepancy score is the percentage of measurements that are not exactly same between subordinate and supervisor. If method is 'percent_non_match', 
                                subordinate_variable and supervisor_variable may be of any datatype.
    n_iterations (int, default 100000): Number of values in null distribution
    ax (plt.axes() instance, default None): axes for plotting null distribution
    
    Outputs:
    discrepancy_scores (array of length n_iterations, float): simulated discrepancy scores with random sub-sampling of measurements
    """
        
    n = len(subordinate_variable)
    discrepancy_scores = []

    for i in tqdm(range(n_iterations)):
        indices = np.random.choice(range(n), size=n, replace=True)
        random_sample_sub = np.array([subordinate_variable[j] for j in indices])
        random_sample_sup = np.array([supervisor_variable[j] for j in indices])

        discrepancy_scores.append(discrepancy_score(random_sample_sub, random_sample_sup, method))
        
    real_discrepancy_score = discrepancy_score(subordinate_variable, supervisor_variable, method)
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(discrepancy_scores, color = 'gray', edgecolor = 'black')
    ax.axvline(real_discrepancy_score, color='r', linestyle='dashed', linewidth=2, label = 'True discrepancy score')
    plt.legend()
    ax.set_xlabel("Discrepancy Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap Distribution of Discrepancy Scores")

    return discrepancy_scores

def shuffle_distribution(subordinate_variable, supervisor_variable, method, n_iterations = 100000, ax = None):
    """Generate a null distribution of discrepancy scores between the two variables 
    by shuffling indices of the supervisor variable.
    
    Inputs:
    subordinate_variable:
    supervisor_variable (array, float): 
    method (string): One of the following options:
        1. 'percent_difference': discrepancy score is the signed difference between subordinate and supervisor measurements, as a percentage of the supervisor measurement
        2. 'absolute_difference': discrepancy score is the absolute difference between subordiante and supervisor measurements
        3. 'absolute_percent_difference': discrepancy score is the absolute difference between subordinate and supervisor measurements, as a percentage of the supervisor measurement
        4. 'simple_difference': discrepancy score is the signed difference between subordinate and supervisor measurements
        5. 'percent_non_match': discrepancy score is the percentage of measurements that are not exactly same between subordinate and supervisor. If method is 'percent_non_match', 
                                subordinate_variable and supervisor_variable may be of any datatype.
    n_iterations (int, default 100000): Number of values in null distribution
    ax (plt.axes() instance, default None): axes for plotting null distribution
    
    Outputs:
    discrepancy_scores (array of length n_iterations, float): simulated discrepancy scores with indices of supervisor variable shuffled 
    
    """
        
    n = len(subordinate_variable)
    discrepancy_scores = []
    subordinate_variable_array = np.array([subordinate_variable[j] for j in range(n)])

    for i in tqdm(range(n_iterations)):
        indices = np.random.choice(range(n), size=n, replace=False)
        random_sample_sup = np.array([supervisor_variable[j] for j in indices])

        discrepancy_scores.append(discrepancy_score(subordinate_variable_array, random_sample_sup, method))
        
    real_discrepancy_score = discrepancy_score(subordinate_variable, supervisor_variable, method)
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(discrepancy_scores, color = 'gray', edgecolor = 'black')
    ax.axvline(real_discrepancy_score, color='r', linestyle='dashed', linewidth=2)
    ax.set_xlabel("Discrepancy Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Shuffled Null Distribution of Discrepancy Scores")

    return discrepancy_scores

def p_value(shuffle_distribution, real_value):
    """Calculate the p-value of the real discrepancy score: 
    the proportion of samples in the shuffled distribution that are less than the real discrepancy score.
    
    Inputs:
    shuffle_distribution (array, float): discrepancy scores simulated under the null hypothesis
    real_value (float): observed discrepancy score
    
    Output:
    p_value (float, between 0 and 1): statistical significance of observed discrepancy score under null hypothesis
    
    """
    
    p_value = sum(shuffle_distribution <= real_value) / len(shuffle_distribution)
    
    return p_value

def teacher_disc_score(row_nos, student_ids, sup_row_nos, sup_student_ids, teacher_constants,
                      teacher_data, sup_data, sup_constants, cb):
    
    teacher_outcomes = []
    sup_outcomes = []

    for sup_row in sup_row_nos: # Iterate over students checked by supervisor for this teacher

        student_id = sup_student_ids[sup_row]
        teacher_row_no = row_nos[student_ids.index(student_id)]

        # Check that subject and class match between the teacher and supervisor
        for c in range(len(teacher_constants)):
            assert(teacher_data[teacher_constants[c]][teacher_row_no] == sup_data[sup_constants[c]][sup_row])

        teacher_score = teacher_data['1st level score'][teacher_row_no] 
        sup_score = sup_data['1st level score'][sup_row] 

        sub = teacher_data['Subject'][teacher_row_no]

        teacher_outcomes.append(teacher_score >= cb[sub])
        sup_outcomes.append(sup_score >= cb[sub])

    disc = disc_score.discrepancy_score(teacher_outcomes, sup_outcomes, 
                                                       method = 'percent_non_match')/100
    return disc