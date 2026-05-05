import numpy as np
from scipy.stats import binom
import matplotlib
import pandas as pd

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import base64
from mpl_toolkits.axes_grid1 import make_axes_locatable
from io import BytesIO

def error_handling(params):
    """
    Perform basic error checks on the input parameters.

    Args:
    params (dict): A dictionary of input parameters.

    Returns:
    tuple: A tuple containing a status code (0 for error, 1 for success) and a message.
    """
    # Basic error checks
    for key, value in params.items():
        if isinstance(value, (int, float)) and value < 0:
            return (0, f"ERROR: {key} must be non-negative")

    if "percent_punish" in params and (
        params["percent_punish"] < 0 or params["percent_punish"] > 100
    ):
        return (0, "ERROR: percent_punish must be between 0 and 100")

    if "percent_guarantee" in params and (
        params["percent_guarantee"] < 0
        or params["percent_guarantee"] > params["percent_punish"]
    ):
        return (0, "ERROR: percent_guarantee must be between 0 and percent_punish")

    if "confidence" in params and (
        params["confidence"] <= 0 or params["confidence"] >= 1
    ):
        return (0, "ERROR: confidence must be between 0 and 1")

    if "distribution" in params and params["distribution"] not in ["uniform", "normal"]:
        return (0, "ERROR: distribution must be 'uniform' or 'normal'")

    if "n_blocks_reward" in params : 
        n_sub_per_block, n_blocks = number_of_subs(
        params["level_test"],
        params["n_subs_per_block"],
        params["n_blocks_per_district"],
        params["n_district"]
        )
        if n_blocks is None:
            return (0, "ERROR: \'level test\' should be either \'Block\' or \'District\' or \'State\''")
        else:
            n_blocks = int(n_blocks)
            if(n_blocks < int(params['n_blocks_reward'])):
                return (0, "ERROR: Number of block rewarded cannot be greater than the number of subjects")

    return (1, "Success")


def number_of_subs(level_test, n_subs_per_block, n_blocks_per_district, n_district):
    """
    Calculate the number of subjects and blocks based on the level of testing.

    Args:
    level_test (str): The level at which the test is being conducted ('Block', 'District', or 'State').
    n_subs_per_block (int): Number of subjects per block.
    n_blocks_per_district (int): Number of blocks per district.
    n_district (int): Number of districts.

    Returns:
    tuple: A tuple containing the number of subjects and the number of blocks.
    """
    if level_test == 'Block':
        return n_subs_per_block, n_blocks_per_district
    elif level_test == 'District':
        return n_subs_per_block * n_blocks_per_district, n_district
    elif level_test == 'State':
        return n_subs_per_block * n_blocks_per_district * n_district, 1
    else:
        print('\'level test\' should be either \'Block\' or \'District\' or \'State\'')
        return None, None

def get_real_ts(n_blocks, average_truth_score, sd_across_blocks, n_sub_per_block, sd_within_block):
    block_mean_ts = generate_true_disc(n_blocks, 0, 1, average_truth_score, sd_across_blocks, 'normal')
    real_order = list(np.argsort(block_mean_ts))
    real_ts = [generate_true_disc(n_sub_per_block, 0, 1, block_mean_ts[block], sd_within_block, 'normal') for block in range(n_blocks)]
    return real_order, real_ts

def get_list_n_sub(n_sub_per_block, min_sub_per_block):
    return list(range(min_sub_per_block, n_sub_per_block + 1))

def get_list_n_samples(total_samples, n_blocks, list_n_sub):
    return [int(total_samples/(n_blocks*n_sub)) for n_sub in list_n_sub]

def get_meas_ts(n_blocks, n_sub_per_block, n_sub_test, n_samples, real_ts):
    meas_ts = np.zeros(n_blocks)
    for block in range(n_blocks):
        subs_test = np.random.choice(list(range(n_sub_per_block)), size=n_sub_test)
        meas_ts[block] = np.mean([binom.rvs(n_samples, real_ts[block][sub])/n_samples for sub in subs_test])
    return meas_ts

def get_ranks(meas_order, real_order, n_blocks, percent_blocks_plot, list_n_sub, n_simulations, errorbar_type):
    n_cond = len(list_n_sub)
    n_blocks_plot = max(1, int(n_blocks*percent_blocks_plot/100))
    
    mean_rank = np.zeros([n_blocks_plot, n_cond])
    errorbars = np.zeros([n_blocks_plot, n_cond])
    
    for block in range(n_blocks_plot):
        for i in range(n_cond):
            ranks = [real_order.index(meas_order[i][n_blocks - block - 1, sim]) + 1 for sim in range(n_simulations)]
            mean_rank[block, i] = np.mean(ranks)
            
            if errorbar_type == 'standard deviation':
                errorbars[block, i] = np.std(ranks, ddof=1)
            elif errorbar_type == 'standard error of the mean':
                errorbars[block, i] = np.std(ranks, ddof=1) / np.sqrt(n_simulations)
            elif errorbar_type == f"95% confidence interval":
                errorbars[block, i] = 1.95 * np.std(ranks, ddof=1) / np.sqrt(n_simulations)
    
    return mean_rank, errorbars

def get_n_blocks_plot(list_n_sub, n_blocks, percent_blocks_plot):
    n_cond = len(list_n_sub)
    n_blocks_plot = max(1, int(n_blocks*percent_blocks_plot/100))
    return n_cond, n_blocks_plot

def get_num_real_units(n_cond, n_simulations, n_blocks_reward, real_order, meas_order, n_blocks, errorbar_type):
    mean_n_real = np.zeros(n_cond)
    errorbars_n_real = np.zeros(n_cond)
    
    for i in range(n_cond):
        n_real = np.zeros(n_simulations)
        for sim in range(n_simulations):
            for block in range(n_blocks_reward):
                # Get real rank of the block with measured rank = n_blocks - block - 1
                real_rank = real_order.index(meas_order[i][n_blocks - block - 1, sim])
                # This block is counted as a 'real' green zone block if its real rank is within the top n_blocks_plot
                if real_rank >= n_blocks - n_blocks_reward:
                    n_real[sim] += 1
        
        mean_n_real[i] = np.mean(n_real)
        if errorbar_type == 'standard deviation':
            errorbars_n_real[i] = np.std(n_real, ddof=1)
        elif errorbar_type == 'standard error of the mean':
            errorbars_n_real[i] = np.std(n_real, ddof=1) / np.sqrt(n_simulations)
        elif errorbar_type == "95% confidence interval":
            errorbars_n_real[i] = 1.95 * np.std(n_real, ddof=1) / np.sqrt(n_simulations)
            
    return mean_n_real, errorbars_n_real

def make_plot(mean_rank, errorbars, list_n_sub, list_n_samples, n_blocks, percent_blocks_plot, errorbar_type):
    fig, ax1 = plt.subplots(figsize=[8, 10])
    n_cond = len(list_n_sub)
    n_blocks_plot = max(1, int(n_blocks * percent_blocks_plot / 100))
    colors = plt.cm.Reds(np.linspace(0.3, 1, n_blocks_plot))

    for block in range(n_blocks_plot):
        ax1.errorbar(list_n_sub, mean_rank[block, :], errorbars[block, :],
                     color=colors[block], marker='o', elinewidth=0.5, capsize=2,
                     label=f'Real rank of unit with measured rank = {n_blocks - block}')

    ax1.plot(list_n_sub, np.ones(n_cond)*n_blocks, color='b', linestyle='--', linewidth=1.5, label='Highest possible rank (k)')
    ax1.legend(fontsize=14, title=f'Errorbars: {errorbar_type}')
    ax1.set_xticks(list_n_sub)
    ax1.set_xlabel('Number of L0s (m) per block tested by supervisor', fontsize=14)
    ax1.set_ylabel('Real rank of blocks with\nthe best measured truth scores', fontsize=14)
    ax1.set_ylim([0, n_blocks + 1])

    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("bottom", size="5%", pad=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_xticks(list_n_sub)
    ax2.set_xticklabels(list_n_samples)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Number of samples (n) per L0', fontsize=14)
    ax2.yaxis.set_visible(False)

    # Chart title
    ax1.set_title('This chart shows the expected outcome for {0} top-scoring units\n selected for reward, using a variety of sampling strategies.\nThe sampling strategy is indicated on the X-axis in terms of the number of L0s\nin each unit tested by a supervisor, and the number of samples tested per L0.\nThe solid lines show the expected ranks of the {0} top-scoring units.\nFor example, if {1} L0s are tested per unit and {2} samples per L0,\nthen the real rank of the unit with the best truth score is expected to be between {3} and {4},\nwith an average expected value of {5}. The dashed blue line shows the best possible rank\nfor any unit (determined by the number of units in the population).\n'.format(
        n_blocks_plot, 
        list_n_sub[0],
        list_n_samples[0], 
        np.round(mean_rank[0, 0] - errorbars[0, 0], 2), 
        np.round(mean_rank[0, 0] + errorbars[0, 0], 2), 
        np.round(mean_rank[0, 0], 2)
        ),
        fontsize=10
    )
    fig.tight_layout(pad=1.0)
    return fig

def get_num_real_units_table(list_n_sub, list_n_samples, mean_n_real, errorbars_n_real, errorbar_type):

    num_real_units_table = pd.DataFrame({'Number of L0s per unit': list_n_sub,
                                 'Number of samples per L0': list_n_samples,
                                 'Number of real units': mean_n_real,
                                 'Errorbar ({0})'.format(errorbar_type): errorbars_n_real
                               })
    return num_real_units_table

def make_plot_num_real_units(list_n_sub, list_n_samples, mean_n_real, errorbars_n_real, n_blocks_plot, errorbar_type, n_blocks, figsize=(8, 11), x_label_fontsize=14, y_label_fontsize=14, linecolor='k', markerstyle='o', elinewidth=0.5, errorbar_capsize=2, legend_fontsize=14):
    """
    Create a matplotlib figure showing the number of 'real' best units found.
    
    Args:
    list_n_sub: List of numbers of subordinates tested per unit
    list_n_samples: List of numbers of samples per subordinate
    mean_n_real: Mean number of real best units found
    errorbars_n_real: Error bars for the number of real best units
    n_blocks_reward: Number of blocks to plot (user input)
    errorbar_type: Type of error bars to display
    
    Returns:
    matplotlib.figure.Figure: The generated figure
    """
    # Create figure and axis handles
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Add text to indicate number of real units on the chart itself, for readability
    plot_height = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    for i in range(len(list_n_sub)):
        if i == len(list_n_sub) - 1:
            y_shift = 0
        elif mean_n_real[i + 1] > mean_n_real[i]:
            y_shift = - 0.02*plot_height*0.2/(mean_n_real[i + 1] - mean_n_real[i])
        elif mean_n_real[i + 1] < mean_n_real[i]:
            y_shift = + 0.02*plot_height*0.2/(mean_n_real[i] - mean_n_real[i + 1])
        elif mean_n_real[i + 1] == mean_n_real[i]:
            y_shift = -0.05*plot_height
        ax1.text(list_n_sub[i] + 0.2*(list_n_sub[1] - list_n_sub[0]), # X axis location of text - slightly to right of plotted point
                 mean_n_real[i] + y_shift, # Y axis location of text - slightly below plotted point
                 np.round(mean_n_real[i], 1), # Text 
                 size = 10)

    # Plot mean and error bars of number of real green zone units
    ax1.errorbar(list_n_sub, mean_n_real, errorbars_n_real, 
                color=linecolor, marker=markerstyle, elinewidth=elinewidth, 
                capsize=errorbar_capsize)

    # Plot dashed line to show the maximum possible number of real green zone units
    ax1.plot(list_n_sub, np.ones(len(list_n_sub))*n_blocks_plot, color='b', linestyle='--', 
             linewidth=1.5, label='Number of units rewarded (b)')
    ax1.legend(fontsize=legend_fontsize, title=f'Errorbars: {errorbar_type}')

    # Set up the primary x-axis (number of L0s per unit)
    ax1.set_xticks(list_n_sub)
    ax1.set_xlim(list_n_sub[0] - 0.5, list_n_sub[-1]*1.1)
    ax1.set_xlabel('Number of L0s tested per unit (m)', fontsize=x_label_fontsize)

    # Create a divider for the primary x-axis to append a new x-axis below
    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("bottom", size="5%", pad=0.7)

    # Hide the new x-axis' frame
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Set the ticks and tick labels for the secondary x-axis
    ax2.set_xticks(list_n_sub)
    ax2.set_xticklabels(list_n_samples)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Number of samples per L0 (n)', fontsize=x_label_fontsize)

    # Hide the y-axis for the secondary x-axis
    ax2.yaxis.set_visible(False)
    ax1.set_ylim([0, n_blocks_plot + 1])

    ax1.set_ylabel("Number of 'real' best units found (c)", fontsize=y_label_fontsize)
    
    # Chart title
    ax1.set_title('This chart shows the expected outcome when {0} out of {1} top-scoring units\nare selected for reward, using a variety of sampling strategies.\nThe sampling strategy is indicated on the X-axis in terms of the number of L0s\nin each unit tested by a supervisor, and the number of samples tested per L0.\nThe solid black line shows how many of the {0} top-scoring units are expected to be\n\'real\' top-scoring units. For example, if {2} L0s are tested per unit\nand {3} samples per L0, then we can be confident that\naround {4} of the {0} rewarded units were deserving of the reward.\nThe dashed blue line shows the number of rewarded units.\n'.format(
        n_blocks_plot, 
        n_blocks, 
        list_n_sub[0], 
        list_n_samples[0], 
        np.round(mean_n_real[0], 1)),
        fontsize=10
    )
    fig.tight_layout(pad=1.0)
    return fig


def generate_true_disc(n, min_disc, max_disc, mean_disc, std_disc, distribution):
    """
    Generate true discrepancy values based on the specified distribution.

    Args:
    n (int): Number of values to generate.
    min_disc (float): Minimum discrepancy value.
    max_disc (float): Maximum discrepancy value.
    mean_disc (float): Mean discrepancy value (for normal distribution).
    std_disc (float): Standard deviation of discrepancy (for normal distribution).
    distribution (str): Type of distribution ('uniform' or 'normal').

    Returns:
    numpy.array: Array of generated discrepancy values.
    """
    if distribution == "uniform":
        return np.random.uniform(min_disc, max_disc, n)
    elif distribution == "normal":
        disc = np.random.normal(mean_disc, std_disc, n)
        return np.clip(disc, min_disc, max_disc)


def generate_meas_disc(true_disc, n_samples):
    """
    Generate measured discrepancy values based on true discrepancy.

    Args:
    true_disc (numpy.array): Array of true discrepancy values.
    n_samples (int): Number of samples per measurement.

    Returns:
    numpy.array: Array of measured discrepancy values.
    """
    return np.array([binom.rvs(n_samples, td) / n_samples for td in true_disc])


def l1_sample_size_calculator(params):
    """
    Calculate the L1 sample size based on given parameters.

    Args:
    params (dict): A dictionary of input parameters.

    Returns:
    dict: A dictionary containing status, message, and calculated sample size.
    """
    error_status, error_message = error_handling(params)
    if error_status == 0:
        return {"status": 0, "message": error_message}

    n_sub = number_of_subs(
        params["level_test"],
        params["n_subs_per_block"],
        params["n_blocks_per_district"],
        params["n_district"],
    )
    n_punish = int(np.ceil((params["percent_punish"] / 100) * n_sub))
    n_guarantee = int(np.ceil((params["percent_guarantee"] / 100) * n_sub))

    def simulate(n_samples):
        true_disc = generate_true_disc(
            n_sub,
            params["min_disc"],
            params["max_disc"],
            params["mean_disc"],
            params["std_disc"],
            params["distribution"],
        )
        meas_disc = generate_meas_disc(true_disc, n_samples)
        worst_offenders = np.argsort(true_disc)[-n_punish:]
        punished = np.argsort(meas_disc)[-n_punish:]
        return len(set(worst_offenders) & set(punished)) >= n_guarantee

    left, right = params["min_n_samples"], params["max_n_samples"]
    while left < right:
        mid = (left + right) // 2
        success_count = sum(simulate(mid) for _ in range(params["n_simulations"]))
        if success_count / params["n_simulations"] >= params["confidence"]:
            right = mid
        else:
            left = mid + 1

    return {
        "status": 1,
        "message": f"L1 sample size calculated successfully.",
        "value": left,
    }


def l2_sample_size_calculator(params):
    """
    Calculate the L2 sample size based on given parameters.

    Args:
    params (dict): A dictionary of input parameters.

    Returns:
    dict: A dictionary containing status, message, and calculated results including true and measured discrepancies.
    """
    error_status, error_message = error_handling(params)
    if error_status == 0:
        return {"status": 0, "message": error_message}

    n_sub = number_of_subs(
        params["level_test"],
        params["n_subs_per_block"],
        params["n_blocks_per_district"],
        params["n_district"],
    )
    n_blocks = n_sub // params["n_subs_per_block"]

    true_disc = generate_true_disc(
        n_blocks,
        0,
        1,
        params["average_truth_score"],
        params["sd_across_blocks"],
        "normal",
    )
    meas_disc = generate_meas_disc(true_disc, params["total_samples"] // n_blocks)

    return {
        "status": 1,
        "message": "L2 sample size calculated successfully.",
        "value": {
            "true_disc": true_disc.tolist(),
            "meas_disc": meas_disc.tolist(),
            "n_samples": params["total_samples"] // n_blocks,
        },
    }

def third_party_sampling_strategy(params):
    error_status, error_message = error_handling(params)
    if error_status == 0:
        return {"status": 0, "message": error_message}

    n_sub_per_block, n_blocks = number_of_subs(
        params["level_test"],
        params["n_subs_per_block"],
        params["n_blocks_per_district"],
        params["n_district"]
    )

    real_order, real_ts = get_real_ts(
        n_blocks,
        params["average_truth_score"],
        params["sd_across_blocks"],
        n_sub_per_block,
        params["sd_within_block"]
    )

    list_n_sub = get_list_n_sub(n_sub_per_block, params["min_sub_per_block"])
    list_n_samples = get_list_n_samples(params["total_samples"], n_blocks, list_n_sub)

    meas_order = {}
    for i, n_sub in enumerate(list_n_sub):
        n_samples = list_n_samples[i]
        meas_order[i] = np.zeros([n_blocks, params["n_simulations"]])
        
        for sim in range(params["n_simulations"]):
            meas_order[i][:, sim] = np.argsort(get_meas_ts(n_blocks, n_sub_per_block, n_sub, n_samples, real_ts))

    mean_rank, errorbars = get_ranks(meas_order, real_order, n_blocks, params["percent_blocks_plot"], list_n_sub, params["n_simulations"], params["errorbar_type"])

    # Get number of 'real' green zone units
    mean_n_real, errorbars_n_real = get_num_real_units(
        len(list_n_sub), 
        params["n_simulations"], 
        params["n_blocks_reward"], 
        real_order, 
        meas_order, 
        n_blocks, 
        params["errorbar_type"]
    )

    # Create first figure (existing plot)
    figImg = make_plot(mean_rank, errorbars, list_n_sub, list_n_samples, n_blocks, params["percent_blocks_plot"], params["errorbar_type"])
    
    # Create second figure using matplotlib
    fig2 = make_plot_num_real_units(
        list_n_sub, 
        list_n_samples, 
        mean_n_real, 
        errorbars_n_real,
        params['n_blocks_reward'],
        params["errorbar_type"],
        n_blocks
    )
    
    # Save both figures to base64
    buf1 = BytesIO()
    figImg.savefig(buf1, format="png")
    plt.close(figImg)
    plot_data1 = base64.b64encode(buf1.getbuffer()).decode("ascii")
    
    buf2 = BytesIO()
    fig2.savefig(buf2, format="png")
    plt.close(fig2)
    plot_data2 = base64.b64encode(buf2.getbuffer()).decode("ascii")

    # Get values of second figure in a pandas dataframe
    second_fig_values = get_num_real_units_table(list_n_sub, list_n_samples, mean_n_real, errorbars_n_real, params["errorbar_type"])
    json_response = second_fig_values.to_dict(orient='records')

    return {
        "status": 1,
        "message": "3P Sampling Strategy calculated successfully.",
        "value": {
            "real_order": [int(x) for x in real_order],
            "meas_order": {str(k): v.tolist() for k, v in meas_order.items()},
            "list_n_sub": [int(x) for x in list_n_sub],
            "list_n_samples": [int(x) for x in list_n_samples],
            "mean_rank": mean_rank.tolist(),
            "errorbars": errorbars.tolist(),
            "mean_n_real": mean_n_real.tolist(),
            "errorbars_n_real": errorbars_n_real.tolist(),
            "figureImg": plot_data1,
            "figure2": plot_data2,
            "table": json_response
        },
    }