import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List
from ranger.metric_containers import AggregatedPairedMetrics


def compute_mean_differences(
    mean_diffs: np.ndarray,
    stdevs_diffs: np.ndarray,
    counts: np.ndarray,
    alpha: float=0.05) -> tuple:
    """
    Compute the raw mean differences, variances, and 
    upper-bounds & lower-bounds of confidence intervals
    given the mean differences and standard deviations 
    of pairwise differences.

    Parameters:
    mean_diffs: A NumPy float array of mean differences
        (mean of the pairwise differences within collections)
    stdevs_diffs: A NumPy float array of standard deviations
        (stdevs of pairwise differences within collections)
    counts: A NumPy int array of number of pairs
        within each collection
    alpha (float): probability of type one error

    Returns:
    tuple: A tuple containing the raw mean differences, variances,
    and upper-bounds & lower-bounds of confidence intervals
    """
    variances = (stdevs_diffs**2) / counts
    standard_errors = variances**(1/2)
    ci_lls, ci_uls, p_values = compute_confidence(
        effect_sizes=mean_diffs,
        standard_errors=standard_errors,
        alpha=alpha
    )

    return mean_diffs, ci_lls, ci_uls, p_values, variances

def compute_standardized_mean_differences(
    mean_diffs: np.ndarray,
    stdevs_diffs: np.ndarray,
    corrs: np.ndarray,
    counts: np.ndarray,
    alpha: float=0.05,
    bias_correction: bool=True) -> tuple:
    """
    Compute the standardized mean differences, variances, and 
    upper-bounds & lower-bounds of confidence intervals
    given the mean differences, standard deviations, and 
    correlations of pairwise differences.

    Parameters:
    mean_diffs: A NumPy float array of mean differences
        (mean of the pairwise differences within collections)
    stds_of_diffs: A NumPy float array of standard deviations
        (stds of pairwise differences within collections)
    corrs: A NumPy float array of correlations
        (correlation between scores within collections)
    counts: A NumPy int array of number of pairs
        within each collection
    alpha (float): probability of type one error

    Returns:
    tuple: A tuple containing the standardized mean differences, variances,
    and upper-bounds & lower-bounds of confidence intervals
    """
    stds_within = stdevs_diffs / (2 * (1-corrs))**(1/2)
    standardized_mean_diffs = mean_diffs / stds_within
    variances = (1/counts + (standardized_mean_diffs**2)/(2*counts)) * 2 * (1-corrs)
    standard_errors = variances**(1/2)
    if bias_correction:
        J = 1 - 3 / (4*(counts-1)-1)
        standardized_mean_diffs *= J
        variances *= J**2
        standard_errors = variances**(1/2)
    ci_lls, ci_uls, p_values = compute_confidence(
        effect_sizes=standardized_mean_diffs,
        standard_errors=standard_errors,
        alpha=alpha
    )

    return standardized_mean_diffs,  ci_lls, ci_uls, p_values, variances

def compute_correlation_effect(
    corrs: np.ndarray,
    counts: np.ndarray,
    alpha: float=0.05) -> tuple:
    """
    Computes correlation effect, confidence interval, and p-value.

    Parameters:
    corrs: A NumPy float array of correlations
    counts: A NumPy int array of sample sizes
      (number of pairs in each collection)
    alpha (float): probability of type one error

    Returns:
    tuple: A tuple containing effect sizes, upper-bounds & lower-bounds of 
    confidence intervals, effect sizes on fischer's z scale and corresponding variances
    """
    # transform to fischers z scale
    z = 0.5 * np.log((1+corrs)/(1-corrs))
    V_z = 1/(counts - 3)
    SE_z = V_z**(1/2)
    # compute confidence intervals in z scale
    ci_lls, ci_uls, p_values = compute_confidence(
        effect_sizes=z,
        standard_errors=SE_z,
        alpha=alpha,
        fishers_z=True
    )

    return corrs, ci_lls, ci_uls, p_values, z, V_z

def combine_effects(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    alpha: float=0.05,
    fishers_z: bool=False) -> tuple:
    """
    Computes the combined effect given effect sizes (mean diff or
    standardized mean diff) and variances of experiments.

    Parameters:
    effect_sizes: A NumPy float array effect sizes (e.g., mean diffs
        or standardized mean diffs of MRRs computed on different collections)
    variances: A NumPy float array of variances
        (variances of pairwise differences, e.g., baseline vs. our model,
        within collections)
    effect_type: A String, indicating the type of the effect. Must be 
        in ["MD", "SMD", "CORR"]
    alpha (float): probability of type one error

    Returns:
    tuple: A tuple summary effect, lower and upper bound of confidence interval,
    p-value (or just bool for sig/not-sig?), weights of the individual experiments
    """
    # estimating between-study variance 
    k = len(effect_sizes)
    df = k - 1
    ## initial weights of the individual experiments are the reciprocal of the variances
    W_i = 1/variances
    Q = np.sum(W_i * effect_sizes**2) - np.sum(W_i * effect_sizes)**2 / np.sum(W_i)
    C = np.sum(W_i) - np.sum(W_i**2) / np.sum(W_i)
    Tsqr = (Q - df) / C
    # estimating summary effect
    V_studies_star = variances + Tsqr
    W_i_star = 1 / V_studies_star
    W_i_star_rel = W_i_star / np.sum(W_i_star)
    M_star = np.sum(W_i_star * effect_sizes) / np.sum(W_i_star)
    V_M_star = 1 / np.sum(W_i_star)
    SE_M_star = V_M_star**(1/2)
    # confidence and sigtest
    ci_lls, ci_uls, p_values = compute_confidence(
        effect_sizes=np.array([M_star]),
        standard_errors=np.array([SE_M_star]),
        alpha=alpha,
        fishers_z=fishers_z
    )
    # convert back if fischer's z scale
    if fishers_z:
        M_star = (np.exp(2*M_star)-1) / (np.exp(2*M_star)+1)

    return M_star, ci_lls[0], ci_uls[0], p_values[0], W_i_star_rel


def compute_confidence(
    effect_sizes: np.ndarray,
    standard_errors: np.ndarray,
    alpha: float=0.05,
    fishers_z: bool=False) -> tuple:
    """
    Computes confidence intervals and p-values given effects and correspoinding
    standard errors, and alpha as input.

    Parameters:
    effect_sizes: A NumPy float array of effect sizes (e.g., mean diffs
        or standardized mean diffs of MRRs computed on different collections)
    standard_errors: A NumPy float array of standard errors
        (variances of pairwise differences, e.g., baseline vs. our model,
        within collections)
    alpha (float): probability of type one error
    fishers_z: A Boolean, indicating if the effects are on z-scale (needs to be 
        on z-scale if effects are correlations)

    Returns:
    tuple: A tuple of np.ndarrays of confidence interval lower & upper limits, 
        and p-values
    """
    # compute confidence intervals
    alpha_z_value = stats.norm.ppf(1-alpha/2)
    ci_lls = effect_sizes - alpha_z_value * standard_errors
    ci_uls = effect_sizes + alpha_z_value * standard_errors
    # if effect-type on fisher z-scale convert back
    if fishers_z:
        ci_lls = (np.exp(2*ci_lls)-1) / (np.exp(2*ci_lls)+1)
        ci_uls = (np.exp(2*ci_uls)-1) / (np.exp(2*ci_uls)+1)
    # compute p value
    Z_values = effect_sizes / standard_errors
    p_values = 2 * (1 - stats.norm.cdf(np.absolute(Z_values)))

    return ci_lls, ci_uls, p_values


def analyze_effects(
    experiment_names: List[str],
    effects: AggregatedPairedMetrics,
    effect_type,
    alpha: float=0.05) -> pd.DataFrame:
    """
    Conducts effect size analysis given paired experiments as input.

    Parameters:
    experiment_names: A List of string experiment names
    effects: An AggregatedPairedMetrics object containing mean differences, 
        standard deviations of paired differences, correlations, and
        sample sizes
    effect_type: A String, indicating the type of the effect. Must be 
        in ["MD", "SMD", "CORR"]
    alpha (float): Probability of type one error

    Returns:
    pd.DataFrame: A pandas dataframe containing effect, lower and upper bound 
        of confidence intervals, relative weights of individual experiments and
        combined effect. 
    """

    experiment_names = [] if experiment_names is None else experiment_names
    summary_df = pd.DataFrame(index=experiment_names)
    mean_diffs = effects.get_mean_diffs()
    stdevs_diffs = effects.get_stdevs_diffs()
    corrs = effects.get_corrs()
    counts = effects.get_counts()

    if effect_type == "MD":
        eff, ci_low, ci_upp, p, var = compute_mean_differences(
            mean_diffs=mean_diffs,
            stdevs_diffs=stdevs_diffs,
            counts=counts,
            alpha=alpha
        )
        comb_eff, comb_ci_low, comb_ci_upp, comb_p, w_re = combine_effects(
            effect_sizes=eff,
            variances=var,
            alpha=alpha
        )
    elif effect_type == "SMD":
        eff, ci_low, ci_upp, p, var = compute_standardized_mean_differences(
            mean_diffs=mean_diffs,
            stdevs_diffs=stdevs_diffs,
            corrs=corrs,
            counts=counts,
            alpha=alpha
        )
        comb_eff, comb_ci_low, comb_ci_upp, comb_p, w_re = combine_effects(
            effect_sizes=eff,
            variances=var,
            alpha=alpha
        )
    elif effect_type == "CORR":
        eff, ci_low, ci_upp, p, z, var_z = compute_correlation_effect(
            corrs=corrs,
            counts=counts,
            alpha=alpha
        )
        comb_eff, comb_ci_low, comb_ci_upp, comb_p, w_re = combine_effects(
            effect_sizes=z,
            variances=var_z,
            alpha=alpha,
            fishers_z=True
        )
    else:
        raise ValueError("Effect type does not exists. Must be in 'MD', 'SMD', or 'CORR'")

    summary_df["eff"] = eff
    summary_df["ci_low"] = ci_low
    summary_df["ci_upp"] = ci_upp
    summary_df["p"] = p
    summary_df["w_re"] = w_re
    # type: ignore
    summary_df.loc["combined_effect"] = [comb_eff, comb_ci_low, comb_ci_upp, comb_p, 1.0]
    return summary_df