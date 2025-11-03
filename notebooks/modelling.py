"""
This file contains the functions for the logistic regression analysis which are used in 4-logistic-regression.ipynb
"""
import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

_SE_PAT = re.compile(r'^se\d+_')
_RE_PAT = re.compile(r'^re\d+_')
_PC_PAT = re.compile(r'^pc\d+_')


def display_name(var: str) -> str:
    if _SE_PAT.match(var):
        return 'self_efficacy'
    if _RE_PAT.match(var):
        return 'response_efficacy'
    if _PC_PAT.match(var):
        return 'perceived_costs'
    return var


def get_variable_lists(config: dict) -> tuple:
    """Extract variable lists from configuration."""
    variables = config['variables']
    threat_vars = variables['threat_appraisal']
    coping_vars = [var for sublist in variables['coping_appraisal'].values()
                   for var in sublist]
    extra_vars = variables['extra_vars']

    return threat_vars, coping_vars, extra_vars


def encode_ordinal_variables(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encode ordinal variables based on configuration mappings."""
    ordinal_mappings = config['ordinal_mappings']
    include_dont_know = config['include_dont_know']
    ordinal_cols = list(ordinal_mappings.keys())

    # Include "Don't know" as a separate category
    if include_dont_know:
        data['perceived_flood_frequency_dont_know'] = np.where(
            data['perceived_flood_frequency'] == "Don't know",
            'Yes', 'No'
        )

    # Select only columns that exist in the data
    existing_ordinal_cols = [
        col for col in ordinal_cols if col in data.columns]

    ordinal_data = data[existing_ordinal_cols].copy()

    ordinal_data_index = ordinal_data.index
    ordinal_data_columns = ordinal_data.columns

    # Apply mappings
    for col in existing_ordinal_cols:
        ordinal_data[col] = ordinal_data[col].map(ordinal_mappings[col])

    # Impute missing values
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    ordinal_encoded = imputer.fit_transform(ordinal_data)

    return pd.DataFrame(ordinal_encoded, columns=ordinal_data_columns, index=ordinal_data_index)


def encode_nominal_variables(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encode nominal variables using one-hot encoding."""
    nominal_cols = config['nominal_cols']
    existing_nominal_cols = [
        col for col in nominal_cols if col in data.columns]

    if not existing_nominal_cols:
        return pd.DataFrame(index=data.index)

    nominal_data = data[existing_nominal_cols].copy()

    encoder = OneHotEncoder(sparse_output=False,
                            drop='first').set_output(transform="pandas")
    nominal_encoded = encoder.fit_transform(nominal_data)

    return nominal_encoded


def prepare_features(data: pd.DataFrame, config: dict, verbose: bool = True) -> pd.DataFrame:
    """Prepare feature matrix X from raw data."""
    threat_vars, coping_vars, extra_vars = get_variable_lists(config)

    # Select variables that exist in the data
    # coping_vars added per structural measure
    selected_vars = threat_vars + extra_vars
    existing_vars = [var for var in selected_vars if var in data.columns]

    X_raw = data[existing_vars].copy()

    # Encode variables
    ordinal_encoded = encode_ordinal_variables(X_raw, config)
    nominal_encoded = encode_nominal_variables(X_raw, config)

    ordinal_encoded_index = ordinal_encoded.index
    nominal_encoded_index = nominal_encoded.index

    # Find overlapping index and subset both dataframes
    overlapping_index = list(set(ordinal_encoded_index)
                             & set(nominal_encoded_index))
    ordinal_encoded = ordinal_encoded.loc[overlapping_index]
    nominal_encoded = nominal_encoded.loc[overlapping_index]

    # Combine encoded features
    X_combined = pd.concat([ordinal_encoded, nominal_encoded], axis=1)

    # Drop rows with missing values
    X_prepared = X_combined.dropna(axis=0)

    if verbose:
        print(
            f"Prepared features: {len(X_prepared)} rows, {len(X_prepared.columns)} columns")
    return X_prepared


def prepare_targets(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare target variables y from raw data."""
    variables = config['variables']
    adaptation_vars = variables['adaptive_behavior']
    y_mappings = config['y_mappings']

    # Select existing adaptation variables
    existing_adaptation_vars = [
        var for var in adaptation_vars if var in data.columns]

    y_raw = data[existing_adaptation_vars].copy()
    y_coded = y_raw.replace(y_mappings)

    print(
        f"Prepared targets: {len(y_coded)} rows, {len(y_coded.columns)} columns")
    return y_coded


def add_structural_measure_features(X_base: pd.DataFrame, data: pd.DataFrame, measure_name: str, measure_num: int, include_all: bool = False) -> pd.DataFrame:
    """Add structural measure-specific features to base feature matrix."""
    if include_all:
        related_vars = [
            f'se{measure_num}_{measure_name}',
            f're{measure_num}_{measure_name}',
            f'pc{measure_num}_{measure_name}'
        ]
    else:
        # Keep the one that does not show high VIF
        related_vars = [
            f'se{measure_num}_{measure_name}'
        ]

    # Only add variables that exist in data
    existing_related_vars = [
        var for var in related_vars if var in data.columns]

    if not existing_related_vars:
        return X_base.copy()

    X_related = data[existing_related_vars].copy()
    X_related = X_related.loc[X_base.index]

    return pd.concat([X_base, X_related], axis=1)


def calculate_vif(X: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """Calculate Variance Inflation Factors for features."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(len(X.columns))]

    # Check for high VIF
    high_vif_features = vif_data.loc[vif_data["VIF"]
                                     > threshold, "feature"].tolist()
    if high_vif_features:
        print(f"Warning: Features with VIF > {threshold}: {high_vif_features}")

    return vif_data


def fit_logistic_model(X_train: pd.DataFrame, y_train: pd.Series) -> sm.Logit | None:
    """Fit logistic regression model using statsmodels."""
    try:
        X_train_sm = sm.add_constant(X_train)
        logit_model = sm.Logit(y_train, X_train_sm).fit(
            cov_type='HC0', disp=0)
        print(logit_model.summary())
        return logit_model
    except Exception as e:
        print(f"Model fitting failed: {e}")
        return None


def extract_significant_variable_stats(model, significance_level: float = 0.05) -> tuple:
    """Extract significant variable statistics from model."""
    pvalues = model.pvalues
    significant_vars = pvalues[pvalues < significance_level].index.tolist()

    # Remove 'const' if present
    significant_vars = [v for v in significant_vars if v != 'const']

    # Calculate odds ratios
    params = model.params
    odds_ratios = np.exp(params)

    # Get confidence intervals and convert to odds ratios
    conf_int = model.conf_int()
    odds_ratios_ci = np.exp(conf_int)

    # Rename variables for interpretability
    renamed_vars = []
    var_odds_ratios = {}
    var_pvalues = {}
    var_confidence_intervals = {}

    patterns = {
        r'^se\d+': 'self_efficacy',
        r'^re\d+': 'response_efficacy',
        r'^pc\d+': 'perceived_costs',
    }

    for var in significant_vars:
        for pat, name in patterns.items():
            if re.match(pat, var):
                renamed_var = name
                break
        else:
            renamed_var = var

        renamed_vars.append(renamed_var)
        var_odds_ratios[renamed_var] = odds_ratios[var] if var in odds_ratios.index else np.nan
        var_pvalues[renamed_var] = pvalues[var] if var in pvalues.index else np.nan

        # Extract confidence intervals (lower, upper)
        if var in odds_ratios_ci.index:
            var_confidence_intervals[renamed_var] = (
                odds_ratios_ci.loc[var, 0], odds_ratios_ci.loc[var, 1])
        else:
            var_confidence_intervals[renamed_var] = (np.nan, np.nan)

    return renamed_vars, var_odds_ratios, var_pvalues, var_confidence_intervals


def extract_all_variable_stats(model, keep_vars=None):
    """Extract all variable statistics from model."""
    params = model.params
    bse = model.bse
    pvals = model.pvalues
    conf = model.conf_int()

    df = pd.DataFrame({
        'coef': params,
        'se': bse,
        'p': pvals,
        'ci_lo': conf[0],
        'ci_hi': conf[1]
    })
    if 'const' in df.index:
        df = df.drop(index='const')

    if keep_vars is not None:
        df = df.loc[[v for v in keep_vars if v in df.index]]

    # convenience: add OR and CI on OR scale
    df['OR'] = np.exp(df['coef'])
    df['OR_lo'] = np.exp(df['ci_lo'])
    df['OR_hi'] = np.exp(df['ci_hi'])
    return df.reset_index(names='variable')


def get_significance_marker_from_p(p_value: float, protective: bool = False) -> str:
    """Get significance marker based on p-value and protective effect."""
    if np.isnan(p_value):
        return ''

    if protective:
        # Use dagger symbol for protective effects
        if p_value < 0.001:
            return '\\textsuperscript{\\dagger\\dagger\\dagger}'
        elif p_value < 0.01:
            return '\\textsuperscript{\\dagger\\dagger}'
        elif p_value < 0.05:
            return '\\textsuperscript{\\dagger}'
        else:
            return ''
    else:
        # Use asterisks for regular effects
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''


def get_significance_marker_from_q(q_value: float, protective: bool = False) -> str:
    """Get significance marker based on q-value and protective effect."""
    if np.isnan(q_value):
        return ''

    if protective:
        if q_value < 0.001:
            return '\\textsuperscript{\\dagger\\dagger\\dagger}'
        elif q_value < 0.01:
            return '\\textsuperscript{\\dagger\\dagger}'
        elif q_value < 0.05:
            return '\\textsuperscript{\\dagger}'
        else:
            return ''
    else:
        if q_value < 0.001:
            return '***'
        elif q_value < 0.01:
            return '**'
        elif q_value < 0.05:
            return '*'
        else:
            return ''


def format_or_ci_with_p(odds_ratio: float, ci_lower: float, ci_upper: float,
                        p_value: float, protective: bool = False) -> str:
    """Format odds ratio with confidence interval and significance marker from p-value."""
    if np.isnan(odds_ratio) or np.isnan(ci_lower) or np.isnan(ci_upper):
        return ''

    # Format OR and CI to 2 decimal places
    or_str = f"{odds_ratio:.2f}"
    ci_str = f"({ci_lower:.2f}–{ci_upper:.2f})"

    # Add significance marker
    sig_marker = get_significance_marker_from_p(p_value, protective)

    return f"{or_str} {ci_str}{sig_marker}"


def format_or_ci_with_q(odds_ratio, ci_lower, ci_upper, q_value, protective=False):
    """Format odds ratio with confidence interval and significance marker from q-value."""
    if np.isnan(odds_ratio) or np.isnan(ci_lower) or np.isnan(ci_upper):
        return ''
    # Format OR and CI to 2 decimal places
    or_str = f"{odds_ratio:.2f}"
    ci_str = f"({ci_lower:.2f}–{ci_upper:.2f})"
    # Add significance marker
    sig_marker = get_significance_marker_from_q(q_value, protective)
    return f"{or_str} {ci_str}{sig_marker}"


def analyze_structural_measure(data: pd.DataFrame, X_base: pd.DataFrame, y_prepared: pd.DataFrame,
                               measure_name: str, measure_num: int, config: dict) -> dict:
    """Analyze a single structural measure."""
    print(f"\\n{'='*60}\\nAnalyzing: {measure_name} (#{measure_num})\\n{'='*60}")

    # Prepare features specific to this structural measure
    X = add_structural_measure_features(
        X_base, data, measure_name, measure_num)

    # Select target variable
    target_col = f'adapt{measure_num}_{measure_name}_agg'

    y = y_prepared[target_col].copy()

    # Exclude "Already implemented" respondents
    y = y[y != 2]
    X = X.loc[y.index]

    print(f"Sample size: {len(X)} (after excluding 'already implemented')")

    # Dummy warning
    if len(X) < 50:  # Minimum sample size check
        print("Warning: Sample size too small for reliable analysis")
        return {}

    # Check multicollinearity
    vif_data = calculate_vif(X, config['vif_threshold'])

    # Regression analysis is performed using statsmodels, which doesn't require train-test split
    X_train = X.astype(float)
    y_train = y.astype(int)

    # Fit model
    model = fit_logistic_model(X_train, y_train)
    if model is None:
        return {}

    # Extract significant variable statistics
    sig_vars, var_odds, var_pvals, var_conf_ints = extract_significant_variable_stats(
        model, config['significance_level'])

    all_vars = extract_all_variable_stats(model)

    # Return results
    return {
        'structural_measure': measure_name,
        'n_train': len(X_train),
        'significant_vars': sig_vars,
        'significant_vars_odds_ratio': var_odds,
        'significant_vars_pvalues': var_pvals,
        'significant_vars_confidence_intervals': var_conf_ints,
        'vif': vif_data,
        'model_summary': model.summary(),
        'full_stats': all_vars
    }


def run_wave_analysis(data: pd.DataFrame, wave_number: int, config: dict | None = None) -> tuple:
    """Run complete analysis for a specific wave."""
    print(f"\\n{'='*80}\\nAnalyzing Wave {wave_number}\\n{'='*80}")

    # Prepare features and targets
    X_base = prepare_features(data, config)
    y_prepared = prepare_targets(data, config)

    # Align X and y indices
    y_prepared = y_prepared.loc[X_base.index]

    # Analyze each structural measure
    results_list = []

    for measure_name, measure_num in config['structural_measures'].items():
        result = analyze_structural_measure(
            data, X_base, y_prepared, measure_name, measure_num, config=config)
        if result:  # Only add if analysis was successful
            results_list.append(result)

    # Create summary DataFrame
    summary_df = create_summary_dataframe(results_list)

    return results_list, summary_df


def create_summary_dataframe(results_list: list) -> pd.DataFrame:
    """Create summary DataFrame from results list."""
    summary_data = []

    for result in results_list:
        summary_data.append({
            'structural_measure': result['structural_measure'],

            'n_significant_vars': len(result['significant_vars']),
            'significant_vars': ', '.join(result['significant_vars']),
            'significant_vars_odds_ratio': result['significant_vars_odds_ratio'],
            'significant_vars_pvalues': result['significant_vars_pvalues']
        })

    return pd.DataFrame(summary_data)


def compare_waves(data: dict[str, pd.DataFrame], wave_numbers: list[int]) -> pd.DataFrame:
    """Compare results across multiple waves."""
    all_summaries = []

    for wave_num in wave_numbers:
        try:
            _, summary_df = run_wave_analysis(data[wave_num], wave_num)
            summary_df['wave'] = wave_num
            all_summaries.append(summary_df)
        except Exception as e:
            print(f"Error analyzing wave {wave_num}: {e}")
            continue

    if not all_summaries:
        return pd.DataFrame()

    combined_df = pd.concat(all_summaries, ignore_index=True)

    # Pivot for easier comparison
    comparison_df = combined_df.pivot_table(
        index='structural_measure',
        columns='wave',
        values=['n_significant_vars'],
        aggfunc='first'
    )

    return comparison_df


def create_significant_variables_matrix(results_by_wave: dict) -> pd.DataFrame:
    """Create matrix of significant variables across waves and measures."""
    all_data = []

    for wave_num, results_list in results_by_wave.items():
        for result in results_list:
            for var in result['significant_vars']:
                all_data.append({
                    'wave': wave_num,
                    'structural_measure': result['structural_measure'],
                    'variable': var,
                    'odds_ratio': result['significant_vars_odds_ratio'].get(var, np.nan),
                    'p_value': result['significant_vars_pvalues'].get(var, np.nan)
                })

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Create pivot table for p-values
    pval_matrix = df.pivot_table(
        index=['wave', 'structural_measure'],
        columns='variable',
        values='p_value',
        aggfunc='first'
    )

    return pval_matrix


def create_odds_ratio_matrix(results_by_wave: dict,
                             protective_threshold: float = 1.0) -> pd.DataFrame:
    """Create matrix of odds ratios with confidence intervals and significance markers across waves and measures.

    Args:
        results_by_wave: Dictionary with wave numbers as keys and results lists as values
        protective_threshold: Threshold below which effects are considered protective (default: 1.0)

    Returns:
        DataFrame with formatted OR (95% CI) values with significance markers
    """
    all_data = []

    for wave_num, results_list in results_by_wave.items():
        for result in results_list:
            for var in result['significant_vars']:
                odds_ratio = result['significant_vars_odds_ratio'].get(
                    var, np.nan)
                p_value = result['significant_vars_pvalues'].get(var, np.nan)
                q_value = result['significant_vars_qvalues'].get(var, np.nan)
                ci_tuple = result['significant_vars_confidence_intervals'].get(
                    var, (np.nan, np.nan))
                ci_lower, ci_upper = ci_tuple

                # Determine if effect is protective
                protective = odds_ratio < protective_threshold if not np.isnan(
                    odds_ratio) else False

                # Format the odds ratio with CI and significance markers
                # formatted_or = format_or_ci_with_p(
                #     odds_ratio, ci_lower, ci_upper, p_value, protective)
                formatted_or = format_or_ci_with_q(
                    odds_ratio, ci_lower, ci_upper, q_value, protective)

                all_data.append({
                    'wave': wave_num,
                    'structural_measure': result['structural_measure'],
                    'variable': var,
                    'formatted_odds_ratio': formatted_or,
                    'raw_odds_ratio': odds_ratio  # Keep for sorting purposes
                })

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Create pivot table for formatted odds ratios
    odds_ratio_matrix = df.pivot_table(
        index=['wave', 'structural_measure'],
        columns='variable',
        values='formatted_odds_ratio',
        aggfunc='first'
    )

    # Create a matrix for raw values to help with sorting
    raw_odds_matrix = df.pivot_table(
        index=['wave', 'structural_measure'],
        columns='variable',
        values='raw_odds_ratio',
        aggfunc='first'
    )

    # Sort columns by number of NaN values (fewer NaNs first) and mean values
    nan_counts = raw_odds_matrix.isna().sum()
    mean_vals = raw_odds_matrix.mean()
    col_sort_df = pd.DataFrame(
        {'nan_count': nan_counts, 'mean_val': mean_vals})
    ordered_cols = col_sort_df.sort_values(
        ['nan_count', 'mean_val'], ascending=[True, False]).index.tolist()
    odds_ratio_matrix = odds_ratio_matrix[ordered_cols]

    return odds_ratio_matrix
