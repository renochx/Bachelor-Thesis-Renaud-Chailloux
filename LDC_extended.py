# Extended model, LDC level, iterative F-tests
import pandas as pd
import os
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.api import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import chi2
from linearmodels.panel import PanelOLS
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Set the working directory
os.chdir('/Users/renaudchailloux/Desktop/Extended Data3')  # Change this to your folder path

# Step 2: Load datasets
def load_datasets():
    recons = pd.read_csv('recons.csv')
    datasets = {
        'VA': pd.read_csv('VA.csv'),  # Voice and Accountability
        'PV': pd.read_csv('PV.csv'),  # Political Stability
        'GE': pd.read_csv('GE.csv'),  # Government Effectiveness
        'RQ': pd.read_csv('RQ.csv'),  # Regulatory Quality
        'RL': pd.read_csv('RL.csv'),  # Rule of Law
        'CC': pd.read_csv('CC.csv'),  # Control of Corruption
        'EA': pd.read_csv('EA.csv'),  # Electricity Access
        'EPC': pd.read_csv('EPC.csv'),  # Electricity Production
        'RD': pd.read_csv('RD.csv'),  # R&D
        'GDP': pd.read_csv('GDP.csv')  # GDP
    }
    return recons, datasets
recons, datasets = load_datasets()

# Step 3: Preprocess datasets
def preprocess_datasets(recons, datasets):
    recons['Estimate'] = pd.to_numeric(recons['Estimate'], errors='coerce')
    for name, dataset in datasets.items():
        if 'Estimate' not in dataset.columns:
            raise KeyError(f"The 'Estimate' column is missing in the {name} dataset.")
        dataset['Estimate'] = pd.to_numeric(dataset['Estimate'], errors='coerce')
        dataset.rename(columns={'Country Name': 'Country', 'Year Column': 'Year', 'Value': 'Estimate'}, inplace=True)
        dataset.rename(columns={'Estimate': f'{name}_Estimate'}, inplace=True)
    return recons, datasets
recons, datasets = preprocess_datasets(recons, datasets)

# Filter datasets to include only Least Developed Countries (LDCs)
ldc_countries = [
   "Afghanistan", "Angola", "Bangladesh", "Benin", "Burkina Faso", "Burundi", "Cambodia", 
   "Central African Republic", "Chad", "Comoros", "Congo, Dem. Rep.", "Djibouti",
   "Eritrea", "Ethiopia", "Gambia, The", "Guinea", "Guinea-Bissau", "Haiti", "Kiribati",
    "Lao PDR", "Lesotho", "Liberia", "Madagascar", "Malawi", "Mali",
    "Mauritania", "Mozambique", "Myanmar", "Nepal", "Niger", "Rwanda", "Senegal", "Sierra Leone",
    "Solomon Islands", "Somalia", "South Sudan", "Sudan", "Timor-Leste", "Togo", "Tuvalu",
    "Tanzania", "Uganda", "Yeme, Rep.", "Zambia",
]
# Filter datasets to include only G20 countries
for name, dataset in datasets.items():
    datasets[name] = dataset[dataset['Country'].isin(ldc_countries)]
# Filter recons to include only G20 countries
recons = recons[recons['Country'].isin(ldc_countries)]

# Step 4: Merge datasets with recons
def merge_datasets(recons, datasets):
    initial_rows = recons.shape[0]
    for name, dataset in datasets.items():
        recons = pd.merge(
            recons,
            dataset,
            on=['Country', 'Year'],
            how='inner',
            suffixes=('', '_other')
        )
        retained_percentage = (recons.shape[0] / initial_rows) * 100
        print(f"After merging with {name}, retained {retained_percentage:.2f}% of rows.")
        if recons.empty:
            raise ValueError(f"Merge with {name} resulted in an empty DataFrame.")
    recons = recons.dropna(subset=['Estimate'])
    print("Final shape of recons after merging all datasets:", recons.shape)
    return recons
recons = merge_datasets(recons, datasets)

# Step 5: Prepare data for regression
def prepare_data(recons, datasets):
    recons = recons.set_index(['Country', 'Year'])
    independent_vars = []
    for name, dataset in datasets.items():
        dataset = dataset.drop_duplicates(subset=['Country', 'Year']).set_index(['Country', 'Year'])
        dataset = dataset.reindex(recons.index)
        independent_vars.append(dataset[f'{name}_Estimate'])
    X = pd.concat(independent_vars, axis=1)
    X.columns = datasets.keys()
    y = recons['Estimate']
    # Align the dependent variable (y) with the independent variables (X)
    # Drop rows with missing values before alignment
    X = X.dropna()
    y = y.dropna()
    #align the indices of X and y
    X, y = X.align(y, join='inner', axis=0)
    # Debugging: Check for missing values
    print("Missing values in X:", X.isnull().sum().sum())
    print("Missing values in y:", y.isnull().sum()) 
    return X, y
X, y = prepare_data(recons, datasets)

# Step 6: Print and save the correlation matrix
def print_and_save_correlation_matrix(X):
    corr_matrix = X.corr()
    print("\nCorrelation Matrix of Independent Variables:\n", corr_matrix)
    # Save to CSV
    corr_matrix.to_csv('/Users/renaudchailloux/Desktop/Output Data/extended_correlation_matrix.csv')
    print("Correlation matrix saved to 'extended_correlation_matrix.csv'")
print_and_save_correlation_matrix(X)

# Step 7: Save descriptive statistics
def save_descriptive_statistics(X, y, recons):
    original_dependent = recons['Estimate']
    combined_data = pd.concat([original_dependent.rename('Dependent_Variable'), X], axis=1)
    descriptive_stats = combined_data.describe()
    descriptive_stats.to_csv('/Users/renaudchailloux/Desktop/Output Data/extended_combined_descriptive_stats.csv')
    print("Descriptive statistics saved to 'combined_descriptive_stats.csv'")
save_descriptive_statistics(X, y, recons)

from linearmodels.panel import PanelOLS, RandomEffects

# Step 8: Scale and log-transform the dependent and independent variables
def scale_and_transform(X, y):
    # Proportionally scale the dependent variable to ensure positivity and apply log transformation
    if y.min() < 1:
        y = y + (1 - y.min())  # Shift so the minimum value becomes 1
    y = np.log(y)  # Apply log transformation

    # Apply log transformation to all independent variables
    for col in X.columns:
        if X[col].min() < 1:
            # Scale proportionally to preserve relative spacing
            X[col] = X[col] + (1 - X[col].min())  # Shift so the minimum value becomes 1
        X[col] = np.log(X[col])  # Apply log transformation to all variables
    # Explicitly return the transformed X and y
    return X, y
# Call the function and unpack the results
X, y = scale_and_transform(X, y)

# Step 9: Add interaction terms
def add_interaction_terms(X):
    print("\nAdding interaction terms between governance indicators and GDP:")
    if 'GDP' not in X.columns:
        raise KeyError("GDP must be included in the dataset to create interaction terms.")
    gdp = X['GDP']
    governance_indicators = ['VA', 'PV', 'GE', 'RQ', 'RL', 'CC']
    for indicator in governance_indicators:
        if indicator in X.columns:
            interaction_term = f"{indicator}_GDP_Interaction"
            X[interaction_term] = X[indicator] * gdp
            print(f"Added interaction term: {interaction_term}")
    return X
X = add_interaction_terms(X)

#Step 10: Fit both models and perform hausman test
#Fit random effects model
def fit_random_effects_model(X, y):
    model = RandomEffects(y, X)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    return results
re_results = fit_random_effects_model(X, y)
print("\nRandom Effects Model Results:")
print(re_results)
# Fit fixed effects model
def fit_fixed_effects_model(X, y):
    model = PanelOLS(y, X, entity_effects=True, time_effects=True, check_rank=True)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    return results
fe_results = fit_fixed_effects_model(X, y)
print("\nFixed Effects Model Results:")
print(fe_results)
def hausman_test(fe_results, re_results, precision=8):
    # Align coefficients
    b_FE = fe_results.params
    b_RE = re_results.params
    common_coef = b_FE.index.intersection(b_RE.index)
    b_FE = b_FE[common_coef]
    b_RE = b_RE[common_coef]
    # Covariance matrices
    v_FE = fe_results.cov.loc[common_coef, common_coef]
    v_RE = re_results.cov.loc[common_coef, common_coef]
    # Compute difference
    diff = b_FE - b_RE
    cov_diff = v_FE - v_RE
    # Handle potential singularity or non-invertible matrix
    try:
        stat = float(diff.T @ np.linalg.inv(cov_diff) @ diff)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix is singular. Hausman test cannot be performed.")
        stat = np.nan
        pval = np.nan
        return {
            "Hausman statistic": stat,
            "degrees of freedom": len(diff),
            "p-value": pval
        }
    # Degrees of freedom
    df = len(diff)
    # P-value
    pval = chi2.sf(stat, df)
    return {
        "Hausman statistic": round(stat, precision),
        "degrees of freedom": df,
        "p-value": round(pval, precision)
    }
hausman_result = hausman_test(fe_results, re_results)
print("Hausman Test Result:")
for k, v in hausman_result.items():
    print(f"{k}: {v}")
hausman_test
# Decide the model type based on the Hausman test
if hausman_result["p-value"] < 0.05:
    print("=> Reject null hypothesis: FE model is preferred (correlation between effects and regressors detected).")
    final_model = PanelOLS(y, X, entity_effects=True, time_effects=True)
else:
    print("=> Fail to reject null hypothesis: RE model is preferred (no evidence of correlation).")
    final_model = RandomEffects(y, X)

# Step 11: Fit the model
def fit_model(X, y):
    model = PanelOLS(y, X, entity_effects=True, time_effects=True, check_rank=True)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    return results
results = fit_model(X, y)

# Step 12: Iteratively perform F-tests and drop insignificant variables
print("\nPerforming F-tests and dropping insignificant variables:")
while True:
    insignificant_vars = []
    for var in X.columns:
         # Drop the variable and fit the reduced model
        X_reduced = X.drop(columns=[var])
        reduced_model = PanelOLS(y, X_reduced, entity_effects=True, time_effects=True, check_rank=True)
        reduced_results = reduced_model.fit()
        # Calculate the F-statistic
        lr_stat = 2 * (results.loglik - reduced_results.loglik)
        df_diff = results.params.shape[0] - reduced_results.params.shape[0]
        p_value = chi2.sf(lr_stat, df_diff)
        print(f"\nVariable: {var}")
        print(f"  F-statistic: {lr_stat}")
        print(f"  P-value: {p_value}")
        print(f"  Degrees of Freedom: {df_diff}")
        # Check significance
        if p_value >= 0.05:
            print(f"  {var} does not contribute significantly to the model (p >= 0.05).")
            insignificant_vars.append(var)
        else:
            print(f"  {var} contributes significantly to the model (p < 0.05).")
    # Break the loop if no insignificant variables are found
    if not insignificant_vars:
        break
    # Drop the insignificant variables
    print(f"\nDropping variable(s): {insignificant_vars}")
    X = X.drop(columns=insignificant_vars)

# Step 13: Refit the final model with only significant variables
print("\nRefitting the final model with significant variables:")
model = PanelOLS(y, X, entity_effects=True, time_effects=True, check_rank=True)
results = model.fit(cov_type='clustered', cluster_entity=True)
print("Final model fitted with significant variables.")

#Step 10: Fit both models and perform hausman test to ensure FE are still warranted 
#Fit random effects model
def fit_random_effects_model(X, y):
    model = RandomEffects(y, X)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    return results
re_results = fit_random_effects_model(X, y)
print("\nRandom Effects Model Results:")
print(re_results)
# Fit fixed effects model
def fit_fixed_effects_model(X, y):
    model = PanelOLS(y, X, entity_effects=True, time_effects=True, check_rank=True)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    return results
fe_results = fit_fixed_effects_model(X, y)
print("\nFixed Effects Model Results:")
print(fe_results)
#Perform 2nd Hausman test
hausman_test
# Decide the model type based on the Hausman test
if hausman_result["p-value"] < 0.05:
    print("=> Reject null hypothesis: FE model is preferred (correlation between effects and regressors detected).")
    final_model = PanelOLS(y, X, entity_effects=True, time_effects=True)
else:
    print("=> Fail to reject null hypothesis: RE model is preferred (no evidence of correlation).")
    final_model = RandomEffects(y, X)

# Step 15: Perform Durbin-Watson test
dw_stat = durbin_watson(results.resids)
print("\nDurbin-Watson Test for Autocorrelation:")
print(f"Durbin-Watson statistic: {dw_stat}")

# Step 16: Perform the Breusch-Pagan test 
# Add a constant column to X
X_with_constant = add_constant(X)
bp_test = het_breuschpagan(results.resids, X_with_constant.values)
bp_stat = bp_test[0]
bp_pvalue = bp_test[1]
f_stat = bp_test[2]
f_pvalue = bp_test[3]
print("\nBreusch-Pagan Test for Heteroskedasticity:")
print(f"LM Statistic: {bp_stat:.4f}")
print(f"LM Test p-value: {bp_pvalue:.4f}")
print(f"F-Statistic: {f_stat:.4f}")
print(f"F-Test p-value: {f_pvalue:.4f}")
if bp_pvalue < 0.05:
    print("Result: Evidence of heteroskedasticity (reject null hypothesis of homoskedasticity).")
else:
    print("Result: No evidence of heteroskedasticity (fail to reject null hypothesis).")

# Step 17: Perform F-test for fixed effects
f_test = results.f_statistic
print("F-test for fixed effects:")
print(f"F-statistic: {f_test.stat}")
print(f"P-value: {f_test.pval}")
print(f"Degrees of freedom: {f_test.df}")
# Determine whether to include fixed effects
if f_test.pval < 0.05:
    print("The F-test indicates that fixed effects are significant. The model should include fixed effects.")
else:
    print("The F-test indicates that fixed effects are not significant. The model should not include fixed effects.")

# Step 18: Save regression results to a CSV file
results_df = results.summary.tables[1].as_html()
results_df = pd.read_html(results_df, header=0, index_col=0)[0]
results_df.to_csv('/Users/renaudchailloux/Desktop/Output Data/Extended Model//LDC_extended_regression_results.csv')
print("Regression results saved to 'LDC_extended_regression_results.csv'")

# Step 19: Save fixed effects
def save_fixed_effects(results):
    fixed_effects = results.estimated_effects
    fe_unstacked = fixed_effects.unstack(level=0)
    fe_unstacked.to_csv('/Users/renaudchailloux/Desktop/Output Data/Extended Model/LDC_extended_unstacked_fixed_effects.csv')
    print("Unstacked fixed effects saved to 'LDC_extended_unstacked_fixed_effects.csv'")
save_fixed_effects(results)
