import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from convert_txt_to_csv import convert_txt_to_csv

# ---------------- Global Configuration ----------------
sns.set_theme(style="whitegrid")
PRIMARY_COLOR = '#1f77b4'       # Muted Blue for data points
SECONDARY_COLOR = '#ff7f0e'     # Muted Orange for analysis region/fit line
ERROR_BAR_COLOR = '#2ca02c'     # Muted Green for error bar points
ERROR_BAR_EDGE_COLOR = '#d62728'  # Muted Red for error bar edges
CONFIDENCE_BAND_COLOR = '#e377c2'  # Muted Pink for confidence bands
GRID_COLOR = '#c7c7c7'          # Light Gray for grid lines

K_BOLTZMANN = 8.617333262e-5  # eV/K
CURRENT = 10e-6  # 10 μA

VOLTAGE_REL_UNC_FACTOR = 0.00015
VOLTAGE_CONST_UNC = 1.5e-4
CURRENT_REL_UNC_FACTOR = 0.00034
CURRENT_CONST_UNC = 200e-9

# ---------------- Helper Functions ----------------
def calculate_error_propagation(temperatures, resistances, mean_inv_temp):
    voltages = resistances * CURRENT
    delta_V = VOLTAGE_REL_UNC_FACTOR * voltages + VOLTAGE_CONST_UNC
    delta_I = CURRENT_REL_UNC_FACTOR * CURRENT + CURRENT_CONST_UNC
    rel_unc_V = delta_V / voltages
    rel_unc_I = delta_I / CURRENT
    # Temperature uncertainty: ΔT = max(2.2, 0.0075*T)
    delta_T = np.maximum(2.2, 0.0075 * temperatures)
    # Propagate error for y = ln(R/T^(3/2))
    delta_y = np.sqrt(rel_unc_V**2 + rel_unc_I**2 + ((9.0/4.0) * (delta_T / temperatures)**2))
    inv_temp = 1 / temperatures
    denominator = np.sum((inv_temp - mean_inv_temp)**2)
    delta_E_g = 2 * K_BOLTZMANN * np.sqrt(np.sum(delta_y**2) / denominator)
    return delta_E_g, delta_y

def plot_analysis_to_axes(file_path, ax_resistance, ax_linear, min_temp=300, max_temp=370, custom_title="Semiconductor Analysis"):
    # Assume the file is CSV-like (or use convert_txt_to_csv beforehand)
    data = pd.read_csv(file_path, comment='#', header=None, names=['Temperature', 'Resistance'])
    data = data.dropna()
    
    # Filter the data
    filtered_data = data[data['Temperature'] >= min_temp].copy()
    if max_temp is not None:
        filtered_data = filtered_data[filtered_data['Temperature'] <= max_temp].copy()
    
    # Create linearized variables
    filtered_data['1/T'] = 1 / filtered_data['Temperature']
    filtered_data['ln(R/T^(3/2))'] = np.log(filtered_data['Resistance'] / (filtered_data['Temperature'] ** (3/2)))
    
    mean_inv_temp = filtered_data['1/T'].mean()
    E_g_error, delta_y = calculate_error_propagation(filtered_data['Temperature'].values,
                                                     filtered_data['Resistance'].values,
                                                     mean_inv_temp)
    filtered_data['err_y'] = delta_y
    delta_T = np.maximum(2.2, 0.0075 * filtered_data['Temperature'])
    filtered_data['err_x'] = delta_T / (filtered_data['Temperature']**2)
    
    # Standard linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_data['1/T'], 
                                                                     filtered_data['ln(R/T^(3/2))'])
    E_g = 2 * K_BOLTZMANN * slope

    # --- Plot 1: Resistance vs Temperature ---
    sns.scatterplot(data=data, x='Temperature', y='Resistance', 
                    color=PRIMARY_COLOR, alpha=0.5, s=30, ax=ax_resistance, label='All Data')
    sns.scatterplot(data=filtered_data, x='Temperature', y='Resistance', 
                    color=SECONDARY_COLOR, s=40, ax=ax_resistance, label=f'T (≥ {min_temp}K)')
    ax_resistance.set_xlabel('Temperature (K)')
    ax_resistance.set_ylabel('Resistance (Ω)')
    ax_resistance.set_title(f"{custom_title} - Resistance")
    ax_resistance.grid(True, color=GRID_COLOR)
    ax_resistance.legend()
    
    # --- Plot 2: Linearized Data and Fit ---
    sns.scatterplot(x=filtered_data['1/T'], y=filtered_data['ln(R/T^(3/2))'],
                    color=PRIMARY_COLOR, s=40, ax=ax_linear, label='Data')
    # Plot error bars on every 5th point
    error_mask = np.zeros(len(filtered_data), dtype=bool)
    error_mask[::5] = True
    error_df = filtered_data[error_mask]
    ax_linear.errorbar(error_df['1/T'], error_df['ln(R/T^(3/2))'], 
                       xerr=error_df['err_x'], yerr=error_df['err_y'],
                       fmt='o', color=ERROR_BAR_COLOR, ecolor=ERROR_BAR_EDGE_COLOR, capsize=3,
                       label='Error')
    # Plot linear fit
    x_fit = np.linspace(filtered_data['1/T'].min(), filtered_data['1/T'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax_linear.plot(x_fit, y_fit, '-', color=SECONDARY_COLOR, linewidth=2, label='Fit')
    
    # 95% confidence band (approximate)
    n = len(filtered_data)
    dof = n - 2
    t_val = stats.t.ppf(0.975, dof)
    y_pred = slope * filtered_data['1/T'] + intercept
    s_err = np.sqrt(np.sum((filtered_data['ln(R/T^(3/2))'] - y_pred)**2) / dof)
    x_mean = np.mean(filtered_data['1/T'])
    s_x = np.sqrt(np.sum((filtered_data['1/T'] - x_mean)**2))
    y_err = t_val * s_err * np.sqrt(1/n + (x_fit - x_mean)**2 / s_x**2)
    ax_linear.fill_between(x_fit, y_fit - y_err, y_fit + y_err, 
                           alpha=0.2, color=CONFIDENCE_BAND_COLOR, label='95% CI')
    
    ax_linear.set_xlabel('1/T (K$^{-1}$)')
    ax_linear.set_ylabel('ln(R/T^(3/2))')
    ax_linear.set_title(f"{custom_title} - Linearized")
    ax_linear.legend()
    ax_linear.grid(True, color=GRID_COLOR)
    
    # Return key results if needed
    return {'E_g': E_g, 'E_g_error': E_g_error, 'slope': slope, 'intercept': intercept, 'r_squared': r_value**2}

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    # Convert your txt files to csv (if needed)
    csv_path1 = convert_txt_to_csv("data_files/feb_13_ntype_run1.txt")
    csv_path2 = convert_txt_to_csv("data_files/feb27_Ge_run2.txt")  # second file
    
    # Create a 2x2 grid: one row per file, left plot is Resistance and right plot is Linearized data.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Analysis for first file (Test 1)
    results1 = plot_analysis_to_axes(csv_path1, axes[0,0], axes[0,1],
                                     min_temp=300, max_temp=370, custom_title="Test 1")
    
    # Analysis for second file (Test 2)
    results2 = plot_analysis_to_axes(csv_path2, axes[1,0], axes[1,1],
                                     min_temp=300, max_temp=370, custom_title="Test 2")
    
    plt.tight_layout()
    plt.show()
    
    # Optionally print out the key analysis results
    print("Results for Test 1:", results1)
    print("Results for Test 2:", results2)
