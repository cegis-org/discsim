import numpy as np
import random
import matplotlib.pyplot as plt
from disc_score import discrepancy_score

def generate_real_scores_per_subject(num_students, mean, std_dev, granularity):
    """
    Generate real test scores for a single subject.
    """
    raw_scores = np.random.normal(loc=mean, scale=std_dev, size=num_students)
    raw_scores = np.clip(raw_scores, 0, 100)  # Ensure scores are between 0 and 100
    quantized_scores = np.round(raw_scores / (100 / (granularity - 1))) * (100 / (granularity - 1))
    return np.clip(quantized_scores, 0, 100)

def generate_real_scores(num_students, subjects_params):
    """
    Generate real test scores for multiple subjects.
    """
    real_scores = {}
    for subject, params in subjects_params.items():
        real_scores[subject] = generate_real_scores_per_subject(
            num_students, params['mean'], params['std_dev'], params['granularity']
        )
    return real_scores

def apply_integrity_distortion(scores, passing_mark, minimum_marks, delta):
    """
    Apply integrity distortion to scores.
    
    Args:
        scores (np.ndarray): Array of real scores.
        passing_mark (float): Passing mark for the subject.
        minimum_marks (float): Minimum marks for the subject.
        delta (float): The marks below the passing mark at which teacher gives passing marks.
    
    Returns:
        np.ndarray: Distorted scores with integrity distortion applied.
    """
    slope = (passing_mark - minimum_marks) / (passing_mark - delta)
    distorted_scores = scores * slope + minimum_marks
    return np.clip(distorted_scores, 0, 100)

def apply_integrity_distortion_L0(real_scores, passing_marks, minimum_marks, delta):
    """
    Apply integrity distortion at L0 for all subjects.
    
    Args:
        real_scores (dict): Dictionary of real scores for each subject.
        passing_marks (dict): Dictionary of passing marks for each subject.
        minimum_marks (dict): Dictionary of minimum marks for each subject.
        delta (float): The marks below the passing mark at which teacher gives passing marks.
    
    Returns:
        dict: Distorted scores with integrity distortion applied at L0.
    """
    distorted_scores = {}
    for subject, scores in real_scores.items():
        passing_mark = passing_marks[subject]
        min_marks = minimum_marks[subject]
        distorted_scores[subject] = apply_integrity_distortion(scores, passing_mark, min_marks, delta)
    return distorted_scores

def apply_integrity_distortion_L1(real_scores, passing_marks, minimum_marks_L0, delta_L0, collusion_index):
    """
    Apply integrity distortion at L1 for all subjects.
    
    Args:
        real_scores (dict): Dictionary of real scores for each subject.
        passing_marks (dict): Dictionary of passing marks for each subject.
        minimum_marks_L0 (dict): Dictionary of L0 minimum marks for each subject.
        delta_L0 (float): The L0 delta value.
        collusion_index (float): Collusion index (0 to 1) for L1 integrity distortion.
    
    Returns:
        dict: Distorted scores with integrity distortion applied at L1.
    """
    if not (0 <= collusion_index <= 1):
        raise ValueError("Collusion index must be between 0 and 1.")
    
    distorted_scores = {}
    for subject, scores in real_scores.items():
        passing_mark = passing_marks[subject]
        min_marks_L1 = minimum_marks_L0[subject] * collusion_index
        delta_L1 = delta_L0 * collusion_index
        distorted_scores[subject] = apply_integrity_distortion(scores, passing_mark, min_marks_L1, delta_L1)
    return distorted_scores

def apply_moderation_distortion(scores, moderation_index):
    """
    Apply moderation distortion to scores.
    
    Args:
        scores (np.ndarray): Array of scores to which moderation distortion will be applied.
        moderation_index (float): Value to be added to the scores as moderation.
    
    Returns:
        np.ndarray: Scores with moderation distortion applied.
    """
    moderated_scores = scores + moderation_index
    return np.clip(moderated_scores, 0, 100)

def apply_measurement_error(scores, mean=0, std_dev=1):
    """
    Apply measurement error to scores.
    
    Args:
        scores (np.ndarray): Array of scores to which measurement error will be applied.
        mean (float, optional): Mean of the normal distribution for measurement error. Default is 0.
        std_dev (float, optional): Standard deviation of the normal distribution for measurement error. Default is 1.
    
    Returns:
        np.ndarray: Scores with measurement error applied.
    """
    noise = np.random.normal(loc=mean, scale=std_dev, size=scores.shape)
    distorted_scores = scores + noise
    return np.clip(distorted_scores, 0, 100)

def apply_distortion_L0(real_scores, passing_marks, minimum_marks, delta, measurement_error_mean=0, measurement_error_std_dev=1):
    """
    Apply all distortions at L0.
    
    Args:
        real_scores (dict): Dictionary of real scores for each subject.
        passing_marks (dict): Dictionary of passing marks for each subject.
        minimum_marks (dict): Dictionary of minimum marks for each subject.
        delta (float): The marks below the passing mark at which teacher gives passing marks.
        measurement_error_mean (float, optional): Mean of the normal distribution for measurement error. Default is 0.
        measurement_error_std_dev (float, optional): Standard deviation of the normal distribution for measurement error. Default is 1.
    
    Returns:
        dict: Distorted scores with all L0 distortions applied.
    """
    distorted_scores = apply_integrity_distortion_L0(real_scores, passing_marks, minimum_marks, delta)
    distorted_scores = {
        subject: apply_measurement_error(scores, mean=measurement_error_mean, std_dev=measurement_error_std_dev)
        for subject, scores in distorted_scores.items()
    }
    return distorted_scores

def apply_distortion_L1(real_scores, passing_marks, minimum_marks_L0, delta_L0, collusion_index, measurement_error_mean=0, measurement_error_std_dev=1, moderation_index_L1=0):
    """
    Apply all distortions at L1.
    
    Args:
        real_scores (dict): Dictionary of real scores for each subject.
        passing_marks (dict): Dictionary of passing marks for each subject.
        minimum_marks_L0 (dict): Dictionary of L0 minimum marks for each subject.
        delta_L0 (float): The L0 delta value.
        collusion_index (float): Collusion index (0 to 1) for L1 integrity distortion.
        measurement_error_mean (float, optional): Mean of the normal distribution for measurement error. Default is 0.
        measurement_error_std_dev (float, optional): Standard deviation of the normal distribution for measurement error. Default is 1.
        moderation_index_L1 (float, optional): Moderation index for L1 distortion. Default is 0.
    
    Returns:
        dict: Distorted scores with all L1 distortions applied.
    """
    distorted_scores = apply_integrity_distortion_L1(real_scores, passing_marks, minimum_marks_L0, delta_L0, collusion_index)
    distorted_scores = {
        subject: apply_moderation_distortion(scores, moderation_index_L1) for subject, scores in distorted_scores.items()
    }
    distorted_scores = {
        subject: apply_measurement_error(scores, mean=measurement_error_mean, std_dev=measurement_error_std_dev)
        for subject, scores in distorted_scores.items()
    }
    return distorted_scores

def apply_distortion_L2(real_scores, measurement_error_mean=0, measurement_error_std_dev=1, moderation_index_L2=0):
    """
    Apply all distortions at L2.
    
    Args:
        real_scores (dict): Dictionary of real scores for each subject.
        measurement_error_mean (float, optional): Mean of the normal distribution for measurement error. Default is 0.
        measurement_error_std_dev (float, optional): Standard deviation of the normal distribution for measurement error. Default is 1.
        moderation_index_L2 (float, optional): Moderation index for L2 distortion. Default is 0.
    
    Returns:
        dict: Distorted scores with all L2 distortions applied.
    """
    distorted_scores = {
        subject: apply_moderation_distortion(scores, moderation_index_L2) for subject, scores in real_scores.items()
    }
    distorted_scores = {
        subject: apply_measurement_error(scores, mean=measurement_error_mean, std_dev=measurement_error_std_dev)
        for subject, scores in distorted_scores.items()
    }
    return distorted_scores

def simulate_test_scores(
    students_per_school, 
    subjects_params, 
    passing_marks, 
    minimum_marks, 
    delta, 
    n_schools_per_L1, 
    n_L1s_per_L2, 
    n_L2s, 
    L1_retest_percentage, 
    L2_retest_percentage_schools, 
    L2_retest_percentage_students, 
    collusion_index, 
    moderation_index_L1=0, 
    moderation_index_L2=0, 
    measurement_error_mean=0, 
    measurement_error_std_dev=1
):
    """
    Simulate test scores through all levels for multiple subjects and hierarchical structure.

    Args:
        students_per_school (int): Number of students in each school.
        subjects_params (dict): Dictionary containing parameters for each subject.
        passing_marks (dict): Dictionary of passing marks for each subject.
        minimum_marks (dict): Dictionary of minimum marks for each subject.
        delta (float): The marks below the passing mark at which teacher gives passing marks.
        n_schools_per_L1 (int): Number of schools grouped into each L1 unit.
        n_L1s_per_L2 (int): Number of L1 units grouped into each L2 unit.
        n_L2s (int): Number of L2 units.
        L1_retest_percentage (float): Percentage of students retested at the L1 level (0 to 100).
        L2_retest_percentage_schools (float): Percentage of schools retested at the L2 level (0 to 100).
        L2_retest_percentage_students (float): Percentage of students retested at the L2 level (0 to 100).
        collusion_index (float): Collusion index for L1 integrity distortion (0 to 1).
        moderation_index_L1 (float, optional): Moderation index for L1 distortion. Default is 0.
        moderation_index_L2 (float, optional): Moderation index for L2 distortion. Default is 0.
        measurement_error_mean (float, optional): Mean of the normal distribution for measurement error. Default is 0.
        measurement_error_std_dev (float, optional): Standard deviation of the normal distribution for measurement error. Default is 1.

    Returns:
        dict: A nested dictionary containing simulated scores organized into L2, L1, and L0 units:
            {
                "L2_<l2_id>": {
                    "L1_<l1_id>": {
                        "school_<school_id>": {
                            "real_scores": Dict of real scores for each student in the school,
                            "L0_scores": Dict of L0 distorted scores for each student in the school,
                            "L1_scores": Dict of L1 distorted scores for retested students in the school,
                            "L2_scores": Dict of L2 distorted scores for retested students in the school
                        },
                        ...
                    },
                    ...
                },
                ...
            }
    """
    # Calculate the total number of schools
    num_schools = n_L2s * n_L1s_per_L2 * n_schools_per_L1

    # Generate unique student IDs
    student_ids = [f"student_{i}" for i in range(num_schools * students_per_school)]

    # Initialize the nested output structure
    nested_scores = {}

    # Generate real scores for all students
    real_scores = {}
    for student_id in student_ids:
        real_scores[student_id] = generate_real_scores(1, subjects_params)

    # Apply L0 distortions and organize by schools (L0 units)
    for l2_index in range(n_L2s):
        l2_key = f"L2_{l2_index}"
        nested_scores[l2_key] = {}

        for l1_index in range(n_L1s_per_L2):
            l1_key = f"L1_{l2_index}_{l1_index}"
            nested_scores[l2_key][l1_key] = {}

            # Get the schools in this L1 unit
            start_school = (l2_index * n_L1s_per_L2 + l1_index) * n_schools_per_L1
            end_school = start_school + n_schools_per_L1

            for school_index in range(start_school, end_school):
                school_key = f"school_{school_index}"
                start_student = school_index * students_per_school
                end_student = start_student + students_per_school
                school_student_ids = student_ids[start_student:end_student]

                # Get real scores and apply L0 distortions for this school
                school_real_scores = {student_id: real_scores[student_id] for student_id in school_student_ids}
                school_L0_scores = {
                    student_id: apply_distortion_L0(
                        real_scores[student_id], 
                        passing_marks, 
                        minimum_marks, 
                        delta, 
                        measurement_error_mean=measurement_error_mean, 
                        measurement_error_std_dev=measurement_error_std_dev
                    )
                    for student_id in school_student_ids
                }

                # Initialize the school dictionary
                nested_scores[l2_key][l1_key][school_key] = {
                    "real_scores": school_real_scores,
                    "L0_scores": school_L0_scores,
                    "L1_scores": {},
                    "L2_scores": {}
                }

    # Apply L1 distortions and organize by L1 units
    for l2_index in range(n_L2s):
        l2_key = f"L2_{l2_index}"

        for l1_index in range(n_L1s_per_L2):
            l1_key = f"L1_{l2_index}_{l1_index}"

            # Get the schools in this L1 unit
            start_school = (l2_index * n_L1s_per_L2 + l1_index) * n_schools_per_L1
            end_school = start_school + n_schools_per_L1

            for school_index in range(start_school, end_school):
                school_key = f"school_{school_index}"
                school_student_ids = list(nested_scores[l2_key][l1_key][school_key]["real_scores"].keys())

                # Select students for L1 retesting
                num_L1_retest = int(len(school_student_ids) * (L1_retest_percentage / 100))
                L1_retest_ids = random.sample(school_student_ids, num_L1_retest)

                # Apply L1 distortions for retested students
                school_L1_scores = {
                    student_id: apply_distortion_L1(
                        real_scores[student_id], 
                        passing_marks, 
                        minimum_marks, 
                        delta, 
                        collusion_index, 
                        measurement_error_mean=measurement_error_mean, 
                        measurement_error_std_dev=measurement_error_std_dev, 
                        moderation_index_L1=moderation_index_L1
                    )
                    for student_id in L1_retest_ids
                }

                # Store L1 scores in the nested structure
                nested_scores[l2_key][l1_key][school_key]["L1_scores"] = school_L1_scores

    # Apply L2 distortions and organize by L2 units
    for l2_index in range(n_L2s):
        l2_key = f"L2_{l2_index}"

        for l1_index in range(n_L1s_per_L2):
            l1_key = f"L1_{l2_index}_{l1_index}"

            # Get the schools in this L1 unit
            start_school = (l2_index * n_L1s_per_L2 + l1_index) * n_schools_per_L1
            end_school = start_school + n_schools_per_L1
            all_schools_in_L1 = list(range(start_school, end_school))

            # Select schools for L2 retesting from this L1 unit
            num_L2_retest_schools = int(len(all_schools_in_L1) * (L2_retest_percentage_schools / 100))
            L2_retest_schools = random.sample(all_schools_in_L1, num_L2_retest_schools)

            # Select students for L2 retesting from the selected schools
            for school_index in L2_retest_schools:
                school_key = f"school_{school_index}"
                school_student_ids = list(nested_scores[l2_key][l1_key][school_key]["L1_scores"].keys())

                # Select a subset of students for L2 retesting
                num_L2_retest_students = int(len(school_student_ids) * (L2_retest_percentage_students / 100))
                L2_retest_ids = random.sample(school_student_ids, num_L2_retest_students)

                # Apply L2 distortions for retested students
                school_L2_scores = {
                    student_id: apply_distortion_L2(
                        real_scores[student_id], 
                        measurement_error_mean=measurement_error_mean, 
                        measurement_error_std_dev=measurement_error_std_dev, 
                        moderation_index_L2=moderation_index_L2
                    )
                    for student_id in L2_retest_ids
                }

                # Store L2 scores in the nested structure
                nested_scores[l2_key][l1_key][school_key]["L2_scores"] = school_L2_scores

    return nested_scores

def plot_nested_scores(nested_scores, subjects, passing_marks):
    """
    Plot the distribution of real scores and compare them with L0, L1, and L2 scores.

    Args:
        nested_scores (dict): The nested dictionary containing scores organized by L2, L1, and schools.
        subjects (list): List of subjects to plot (e.g., ["Maths", "English", "Science"]).
        passing_marks (dict): Dictionary of passing marks for each subject.
    """
    # Collect all scores for each subject
    real_scores = {subject: [] for subject in subjects}
    L0_scores = {subject: [] for subject in subjects}
    L1_real_scores = {subject: [] for subject in subjects}
    L1_scores = {subject: [] for subject in subjects}
    L2_real_scores = {subject: [] for subject in subjects}
    L2_scores = {subject: [] for subject in subjects}

    # Traverse the nested_scores dictionary to extract scores
    for l2_data in nested_scores.values():
        for l1_data in l2_data.values():
            for school_data in l1_data.values():
                # Add real scores
                for student_scores in school_data["real_scores"].values():
                    for subject in subjects:
                        real_scores[subject].extend(student_scores[subject])  # Use extend to flatten arrays

                # Add L0 scores
                for student_scores in school_data["L0_scores"].values():
                    for subject in subjects:
                        L0_scores[subject].extend(student_scores[subject])  # Use extend to flatten arrays

                # Add L1 scores and corresponding real scores
                for student_id, student_scores in school_data["L1_scores"].items():
                    for subject in subjects:
                        L1_real_scores[subject].append(school_data["real_scores"][student_id][subject])
                        L1_scores[subject].append(student_scores[subject])

                # Add L2 scores and corresponding real scores
                for student_id, student_scores in school_data["L2_scores"].items():
                    for subject in subjects:
                        L2_real_scores[subject].append(school_data["real_scores"][student_id][subject])
                        L2_scores[subject].append(student_scores[subject])

    # Plot the distributions and comparisons
    num_subjects = len(subjects)
    fig, axes = plt.subplots(4, num_subjects, figsize=(5 * num_subjects, 20))

    # Font size settings
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12

    for i, subject in enumerate(subjects):
        passing_mark = passing_marks[subject]
        # Plot histogram of real scores
        axes[0, i].hist(real_scores[subject], bins=20, color="black", alpha=0.7)
        axes[0, i].set_title(f"Real Scores Distribution - {subject}", fontsize=title_fontsize)
        axes[0, i].set_xlabel("Score", fontsize=label_fontsize)
        axes[0, i].set_ylabel("Frequency", fontsize=label_fontsize)
        axes[0, i].tick_params(axis="both", labelsize=tick_fontsize)

        # Scatter plot: Real vs L0 scores
        axes[1, i].scatter(real_scores[subject], L0_scores[subject], alpha=0.5, color="black")
        # Vertical line showing passing marks
        axes[1, i].axvline(passing_mark, color="red", linestyle="--", label="Passing Mark")
        # Horizontal line showing passing marks
        axes[1, i].axhline(passing_mark, color="red", linestyle="--", label="Passing Mark")
        axes[1, i].set_title(f"Real vs L0 Scores - {subject}", fontsize=title_fontsize)
        axes[1, i].set_xlabel("Real Scores", fontsize=label_fontsize)
        axes[1, i].set_ylabel("L0 Scores", fontsize=label_fontsize)
        axes[1, i].tick_params(axis="both", labelsize=tick_fontsize)
        axes[1, i].set_xlim(-5, 105)
        axes[1, i].set_ylim(-5, 105)
        axes[1, i].grid()

        # Scatter plot: Real vs L1 scores
        axes[2, i].scatter(L1_real_scores[subject], L1_scores[subject], alpha=0.5, color="black")
        # Vertical line showing passing marks
        axes[2, i].axvline(passing_mark, color="red", linestyle="--", label="Passing Mark")
        # Horizontal line showing passing marks
        axes[2, i].axhline(passing_mark, color="red", linestyle="--", label="Passing Mark")
        axes[2, i].set_title(f"Real vs L1 Scores - {subject}", fontsize=title_fontsize)
        axes[2, i].set_xlabel("Real Scores (L1 Retested)", fontsize=label_fontsize)
        axes[2, i].set_ylabel("L1 Scores", fontsize=label_fontsize)
        axes[2, i].tick_params(axis="both", labelsize=tick_fontsize)
        axes[2, i].set_xlim(-5, 105)
        axes[2, i].set_ylim(-5, 105)
        axes[2, i].grid()

        # Scatter plot: Real vs L2 scores
        axes[3, i].scatter(L2_real_scores[subject], L2_scores[subject], alpha=0.5, color="black")
        # Vertical line showing passing marks
        axes[3, i].axvline(passing_mark, color="red", linestyle="--", label="Passing Mark")
        # Horizontal line showing passing marks
        axes[3, i].axhline(passing_mark, color="red", linestyle="--", label="Passing Mark")
        axes[3, i].set_title(f"Real vs L2 Scores - {subject}", fontsize=title_fontsize)
        axes[3, i].set_xlabel("Real Scores (L2 Retested)", fontsize=label_fontsize)
        axes[3, i].set_ylabel("L2 Scores", fontsize=label_fontsize)
        axes[3, i].tick_params(axis="both", labelsize=tick_fontsize)
        axes[3, i].set_xlim(-5, 105)
        axes[3, i].set_ylim(-5, 105)
        axes[3, i].grid()

    plt.tight_layout()
    plt.show()

def calculate_disc_scores(nested_scores, method, passing_marks, subjects):
    """
    Calculate discrepancy scores for three pairs of scores: L0 vs L2, L1 vs L2, and L0 vs L1.
    For each L0 (school), calculate the discrepancy score for L0 vs L2 and L0 vs L1.
    For each L1 (unit), calculate the discrepancy score for L1 vs L2.
    Plot the distributions of discrepancy scores for each pair.

    Args:
        nested_scores (dict): The nested dictionary containing scores organized by L2, L1, and schools.
        method (str): The method to calculate discrepancy scores (e.g., 'percent_difference', 'absolute_difference', etc.).
        passing_marks (dict): Dictionary of passing marks for each subject.

    Returns:
        dict: A dictionary containing arrays of discrepancy scores for each pair.
    """
    # Initialize arrays to store discrepancy scores
    L0_vs_L2_scores = []
    L1_vs_L2_scores = []
    L0_vs_L1_scores = []

    # Helper function to binarize scores into pass/fail
    def binarize_scores(scores, passing_marks):
        binarized = []
        for subject, score in scores.items():
            binarized.append(score >= passing_marks[subject])
        return binarized

    # Traverse the nested_scores dictionary to calculate discrepancy scores
    for l2_key, l2_data in nested_scores.items():
        for l1_key, l1_data in l2_data.items():
            # Collect L1 vs L2 scores for this L1 unit
            L1_subordinate = []
            L2_supervisor = []

            for school_key, school_data in l1_data.items():
                # Collect L0 vs L2 scores for this school
                if school_data["L2_scores"]:  # Only calculate if L2 retested this school
                    L0_subordinate = []
                    L2_supervisor_school = []

                    for student_id in school_data["L2_scores"]:
                        if student_id in school_data["L0_scores"]:
                            if method in ["percent_non_match", "directional_percent_non_match"]:
                                L0_subordinate.extend(binarize_scores(school_data["L0_scores"][student_id], passing_marks))
                                L2_supervisor_school.extend(binarize_scores(school_data["L2_scores"][student_id], passing_marks))
                            else:
                                L0_subordinate.extend(school_data["L0_scores"][student_id].values())
                                L2_supervisor_school.extend(school_data["L2_scores"][student_id].values())

                    if L0_subordinate and L2_supervisor_school:
                        L0_vs_L2_scores.append(discrepancy_score(L0_subordinate, L2_supervisor_school, method))

                # Collect L0 vs L1 scores for this school
                if school_data["L1_scores"]:  # Only calculate if L1 retested this school
                    L0_subordinate = []
                    L1_supervisor_school = []

                    for student_id in school_data["L1_scores"]:
                        if student_id in school_data["L0_scores"]:
                            if method in ["percent_non_match", "directional_percent_non_match"]:
                                L0_subordinate.extend(binarize_scores(school_data["L0_scores"][student_id], passing_marks))
                                L1_supervisor_school.extend(binarize_scores(school_data["L1_scores"][student_id], passing_marks))
                            else:
                                L0_subordinate.extend(school_data["L0_scores"][student_id].values())
                                L1_supervisor_school.extend(school_data["L1_scores"][student_id].values())

                    if L0_subordinate and L1_supervisor_school:
                        L0_vs_L1_scores.append(discrepancy_score(L0_subordinate, L1_supervisor_school, method))

                # Collect L1 vs L2 scores for this L1 unit
                if school_data["L2_scores"]:  # Only calculate if L2 retested this school
                    for student_id in school_data["L2_scores"]:
                        if student_id in school_data["L1_scores"]:
                            if method in ["percent_non_match", "directional_percent_non_match"]:
                                L1_subordinate.extend(binarize_scores(school_data["L1_scores"][student_id], passing_marks))
                                L2_supervisor.extend(binarize_scores(school_data["L2_scores"][student_id], passing_marks))
                            else:
                                L1_subordinate.extend(school_data["L1_scores"][student_id].values())
                                L2_supervisor.extend(school_data["L2_scores"][student_id].values())

            # Calculate L1 vs L2 discrepancy for this L1 unit
            if L1_subordinate and L2_supervisor:
                L1_vs_L2_scores.append(discrepancy_score(L1_subordinate, L2_supervisor, method))

    # Store the results in a dictionary
    results = {
        "L0_vs_L2": L0_vs_L2_scores,
        "L1_vs_L2": L1_vs_L2_scores,
        "L0_vs_L1": L0_vs_L1_scores
    }

    # Determine the global x-axis limits
    all_scores = L0_vs_L2_scores + L1_vs_L2_scores + L0_vs_L1_scores
    x_min, x_max = min(all_scores), max(all_scores)

    # Plot the distributions of discrepancy scores
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Font size settings
    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 16

    # Helper function to plot histograms as percentages
    def plot_histogram(ax, data, title):
        counts, bins, patches = ax.hist(data, bins=20, range=(x_min, x_max), color="black", alpha=0.7, density=True)
        percentages = counts * 100  # Convert to percentages
        ax.clear()
        ax.bar(bins[:-1], percentages, width=np.diff(bins), align="edge", color="black", alpha=0.7)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel("Discrepancy Score", fontsize=label_fontsize)
        ax.set_ylabel("Percentage", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    # L0 vs L2
    plot_histogram(axes[0], L0_vs_L2_scores, "L0 vs L2 Discrepancy")

    # L1 vs L2
    plot_histogram(axes[1], L1_vs_L2_scores, "L1 vs L2 Discrepancy")

    # L0 vs L1
    plot_histogram(axes[2], L0_vs_L1_scores, "L0 vs L1 Discrepancy")

    # Combined scatter plot for L0 vs L1 and L1 vs L2 discrepancies versus real scores
    fig, axes = plt.subplots(2, len(subjects), figsize=(6 * len(subjects), 12))

    # Ensure axes is a 2D array even if there's only one subject
    if len(subjects) == 1:
        axes = np.array([axes]).T

    # Iterate over each subject
    for i, subject in enumerate(subjects):
        real_scores_for_L0_vs_L1 = []
        discrepancies_for_L0_vs_L1 = []
        real_scores_for_L1_vs_L2 = []
        discrepancies_for_L1_vs_L2 = []

        # Traverse the nested_scores dictionary to calculate discrepancies for the current subject
        for l2_data in nested_scores.values():
            for l1_data in l2_data.values():
                for school_data in l1_data.values():
                    for student_id in school_data["real_scores"]:
                        # Check if the student has scores for L0, L1, and L2 for the current subject
                        if (
                            subject in school_data["real_scores"][student_id]
                            and subject in school_data["L0_scores"].get(student_id, {})
                            and subject in school_data["L1_scores"].get(student_id, {})
                        ):
                            # Extract real score and calculate L0 vs L1 discrepancy
                            real_score = school_data["real_scores"][student_id][subject]
                            real_scores_for_L0_vs_L1.append(real_score)
                            L0_score = school_data["L0_scores"][student_id][subject]
                            L1_score = school_data["L1_scores"][student_id][subject]
                            discrepancy_L0_L1 = discrepancy_score([L0_score], [L1_score], method)
                            discrepancies_for_L0_vs_L1.append(discrepancy_L0_L1)

                        # Check if the student has scores for L1 and L2 for the current subject
                        if (
                            subject in school_data["real_scores"][student_id]
                            and subject in school_data["L1_scores"].get(student_id, {})
                            and subject in school_data["L2_scores"].get(student_id, {})
                        ):
                            # Extract real score and calculate L1 vs L2 discrepancy
                            real_score = school_data["real_scores"][student_id][subject]
                            real_scores_for_L1_vs_L2.append(real_score)
                            L1_score = school_data["L1_scores"][student_id][subject]
                            L2_score = school_data["L2_scores"][student_id][subject]
                            discrepancy_L1_L2 = discrepancy_score([L1_score], [L2_score], method)
                            discrepancies_for_L1_vs_L2.append(discrepancy_L1_L2)

        # Plot L0 vs L1 discrepancy for the current subject
        axes[0, i].scatter(real_scores_for_L0_vs_L1, discrepancies_for_L0_vs_L1, alpha=0.5, color="black")
        axes[0, i].set_title(f"{subject} (L0 vs L1)", fontsize=title_fontsize)
        axes[0, i].set_xlabel("Real Scores", fontsize=label_fontsize)
        axes[0, i].set_ylabel("L0 vs L1 Discrepancy\nstudent-wise", fontsize=label_fontsize)
        axes[0, i].tick_params(axis="both", labelsize=tick_fontsize)
        axes[0, i].grid()

        # Plot L1 vs L2 discrepancy for the current subject
        axes[1, i].scatter(real_scores_for_L1_vs_L2, discrepancies_for_L1_vs_L2, alpha=0.5, color="black")
        axes[1, i].set_title(f"{subject} (L1 vs L2)", fontsize=title_fontsize)
        axes[1, i].set_xlabel("Real Scores", fontsize=label_fontsize)
        axes[1, i].set_ylabel("L1 vs L2 Discrepancy\nstudent-wise", fontsize=label_fontsize)
        axes[1, i].tick_params(axis="both", labelsize=tick_fontsize)
        axes[1, i].grid()

    plt.tight_layout()
    plt.show()

    return results



