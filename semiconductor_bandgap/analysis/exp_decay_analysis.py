import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from convert_txt_to_csv import convert_txt_to_csv

# =================== Global Configuration ===================

# Plot style and colors
sns.set_theme(style="whitegrid")
PRIMARY_COLOR = '#1f77b4'       # Muted Blue for all data points
SECONDARY_COLOR = '#ff7f0e'     # Muted Orange for analysis region and fit line
ERROR_BAR_COLOR = '#2ca02c'     # Muted Green for points with error bars
ERROR_BAR_EDGE_COLOR = '#d62728'  # Muted Red for error bar edges
ERROR_LINE_COLOR = '#9467bd'    # Muted Purple for error lines
RESIDUAL_COLOR = '#8c564b'      # Muted Brown for residuals
CONFIDENCE_BAND_COLOR = '#e377c2'  # Muted Pink for confidence bands
TEXT_BOX_COLOR = '#7f7f7f'      # Gray for text boxes
GRID_COLOR = '#c7c7c7'          # Light Gray for grid lines

# Analysis and file configuration
DATA_TXT_PATH_1 = "data_files/feb_13_ntype_run1.txt"  # First file
DATA_TXT_PATH_2 = "data_files/feb27_Ge_run1.txt"  # Second file
CUSTOM_TITLE_1 = "Silicon (n-type) Sample Test"      # Title for first file
CUSTOM_TITLE_2 = "Germanium Sample Test"      # Title for second file
# Temperature ranges for analysis (separate for each file):
MIN_TEMP_1 = 520
MAX_TEMP_1 = 580
MIN_TEMP_2 = 300
MAX_TEMP_2 = 370

# Physical constants and measurement settings
K_BOLTZMANN = 8.617333262e-5  # eV/K
CURRENT = 10e-6  # 10 μA

# Uncertainty parameters (all in SI units)
VOLTAGE_REL_UNC_FACTOR = 0.00015
VOLTAGE_CONST_UNC = 1.5e-4      # Use the same constant in all parts of the analysis
CURRENT_REL_UNC_FACTOR = 0.00034
CURRENT_CONST_UNC = 200e-9

# =================== Function Definitions ===================

def calculate_error_propagation(temperatures, resistances, mean_inv_temp):
    """
    Calculate the uncertainty in bandgap energy using error propagation.
    """
    # Compute measured voltage (V = R * I)
    voltages = resistances * CURRENT

    # Voltage and current uncertainties using global constants
    delta_V = VOLTAGE_REL_UNC_FACTOR * voltages + VOLTAGE_CONST_UNC
    delta_I = CURRENT_REL_UNC_FACTOR * CURRENT + CURRENT_CONST_UNC
    
    rel_unc_V = delta_V / voltages
    rel_unc_I = delta_I / CURRENT
    
    # Temperature uncertainty: ΔT = max(2.2, 0.0075*T)
    delta_T = np.maximum(2.2, 0.0075 * temperatures)
    
    # For y = ln(R/T^(3/2)), propagate errors:
    delta_y = np.sqrt(rel_unc_V**2 + rel_unc_I**2 + ((9.0/4.0) * (delta_T / temperatures)**2))
    
    # For the independent variable x = 1/T
    inv_temp = 1 / temperatures
    denominator = np.sum((inv_temp - mean_inv_temp)**2)
    
    # Propagate error to E_g:
    delta_E_g = 2 * K_BOLTZMANN * np.sqrt(np.sum(delta_y**2) / denominator)
    
    return delta_E_g, delta_y

def analyze_semiconductor_bandgap(file_path, min_temp=500, max_temp=None, custom_title=None):
    """
    Analyze semiconductor bandgap energy from temperature-resistance data with error analysis.
    Returns data and results for plotting later.
    """
    # Read and parse the file manually (ignoring header lines)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data_lines = []
    for line in lines:
        if line.strip() and not line.startswith('Semiconductor') and not line.startswith('Temperature'):
            data_lines.append(line.strip())
    
    temperatures = []
    resistances = []
    for line in data_lines:
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                temperatures.append(float(parts[0]))
                resistances.append(float(parts[1]))
            except ValueError:
                print(f"Warning: Could not parse line: {line}")
    
    data = pd.DataFrame({
        'Temperature': temperatures,
        'Resistance': resistances
    })
    
    # Filter data by temperature
    filtered_data = data[data['Temperature'] >= min_temp].copy()
    if max_temp is not None:
        filtered_data = filtered_data[filtered_data['Temperature'] <= max_temp].copy()
    
    # Create linearized variables
    filtered_data['1/T'] = 1 / filtered_data['Temperature']
    filtered_data['ln(R/T^(3/2))'] = np.log(filtered_data['Resistance'] / (filtered_data['Temperature'] ** (3/2)))
    
    mean_inv_temp = filtered_data['1/T'].mean()
    E_g_error, delta_y = calculate_error_propagation(
        filtered_data['Temperature'].values,
        filtered_data['Resistance'].values,
        mean_inv_temp
    )
    filtered_data['err_y_propagated'] = delta_y
    
    # Calculate error in x = 1/T
    delta_T = np.maximum(2.2, 0.0075 * filtered_data['Temperature'])
    filtered_data['err_x'] = delta_T / (filtered_data['Temperature']**2)
    
    # Standard (unweighted) linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered_data['1/T'], filtered_data['ln(R/T^(3/2))']
    )
    E_g = 2 * K_BOLTZMANN * slope  # in eV
    E_g_std_err = 2 * K_BOLTZMANN * std_err
    
    # Calculate predictions and residuals
    y_pred = slope * filtered_data['1/T'] + intercept
    residuals = filtered_data['ln(R/T^(3/2))'] - y_pred
    n = len(filtered_data)
    dof = n - 2
    chi_squared = np.sum((residuals / filtered_data['err_y_propagated'])**2)
    reduced_chi_squared = chi_squared / dof
    
    # --- Weighted Linear Fit ---
    weights = 1.0 / (filtered_data['err_y_propagated']**2)
    p, cov = np.polyfit(filtered_data['1/T'], filtered_data['ln(R/T^(3/2))'], deg=1, w=np.sqrt(weights), cov=True)
    weighted_slope = p[0]
    weighted_intercept = p[1]
    weighted_E_g = 2 * K_BOLTZMANN * weighted_slope
    weighted_E_g_err = 2 * K_BOLTZMANN * np.sqrt(cov[0, 0])
    y_weighted_pred = weighted_slope * filtered_data['1/T'] + weighted_intercept
    weighted_r_squared = 1 - np.sum((filtered_data['ln(R/T^(3/2))'] - y_weighted_pred)**2) / np.sum((filtered_data['ln(R/T^(3/2))'] - np.mean(filtered_data['ln(R/T^(3/2))']))**2)
    
    # Calculate error contributions using global uncertainty parameters
    voltages = filtered_data['Resistance'] * CURRENT
    delta_V = VOLTAGE_REL_UNC_FACTOR * voltages + VOLTAGE_CONST_UNC
    delta_I = CURRENT_REL_UNC_FACTOR * CURRENT + CURRENT_CONST_UNC
    delta_T = np.maximum(2.2, 0.0075 * filtered_data['Temperature'])
    rel_unc_V = delta_V / voltages
    rel_unc_I = delta_I / CURRENT
    rel_unc_T = delta_T / filtered_data['Temperature']
    V_contrib = rel_unc_V**2
    I_contrib = rel_unc_I**2
    T_contrib = (9.0/4.0) * rel_unc_T**2
    total_contrib = V_contrib + I_contrib + T_contrib
    filtered_data['V_contrib_pct'] = 100 * V_contrib / total_contrib
    filtered_data['I_contrib_pct'] = 100 * I_contrib / total_contrib
    filtered_data['T_contrib_pct'] = 100 * T_contrib / total_contrib
    
    results = {
        'data': data,
        'filtered_data': filtered_data,
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'bandgap_energy_eV': E_g,
        'bandgap_error_eV': E_g_error,
        'bandgap_std_err_eV': E_g_std_err,
        'r_squared': r_value**2,
        'reduced_chi_squared': reduced_chi_squared,
        'data_points_used': n,
        'residuals': residuals,
        'weighted_bandgap_energy_eV': weighted_E_g,
        'weighted_bandgap_error_eV': weighted_E_g_err,
        'weighted_r_squared': weighted_r_squared,
        'V_contrib_pct_mean': filtered_data['V_contrib_pct'].mean(),
        'I_contrib_pct_mean': filtered_data['I_contrib_pct'].mean(),
        'T_contrib_pct_mean': filtered_data['T_contrib_pct'].mean()
    }
    
    return results

def plot_2x2_comparison(results1, results2, title1, title2, min_temp_1, max_temp_1, min_temp_2, max_temp_2):
    """
    Plot a 2x2 comparison of analyses from two different files.
    """
    # Create the 2x2 subplot configuration with additional bottom space for summary
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(bottom=0.15)  # Add extra space at the bottom
    
    # Top left: Resistance vs Temperature for file 1
    data1 = results1['data']
    filtered_data1 = results1['filtered_data']
    sns.scatterplot(data=data1, x='Temperature', y='Resistance',
                    color=PRIMARY_COLOR, alpha=0.5, s=30, label='All Data', ax=axs[0, 0])
    analysis_label1 = f'Analysis Region ({min_temp_1}K ≤ T ≤ {max_temp_1}K)'
    sns.scatterplot(data=filtered_data1, x='Temperature', y='Resistance',
                    color=SECONDARY_COLOR, s=40, label=analysis_label1, ax=axs[0, 0])
    axs[0, 0].set_xlabel('Temperature (K)')
    axs[0, 0].set_ylabel('Resistance (Ω)')
    axs[0, 0].set_title(f"{title1} - Resistance vs Temperature")
    axs[0, 0].grid(True, color=GRID_COLOR)
    axs[0, 0].legend()
    
    # Top right: Resistance vs Temperature for file 2
    data2 = results2['data']
    filtered_data2 = results2['filtered_data']
    sns.scatterplot(data=data2, x='Temperature', y='Resistance',
                    color=PRIMARY_COLOR, alpha=0.5, s=30, label='All Data', ax=axs[0, 1])
    analysis_label2 = f'Analysis Region ({min_temp_2}K ≤ T ≤ {max_temp_2}K)'
    sns.scatterplot(data=filtered_data2, x='Temperature', y='Resistance',
                    color=SECONDARY_COLOR, s=40, label=analysis_label2, ax=axs[0, 1])
    axs[0, 1].set_xlabel('Temperature (K)')
    axs[0, 1].set_ylabel('Resistance (Ω)')
    axs[0, 1].set_title(f"{title2} - Resistance vs Temperature")
    axs[0, 1].grid(True, color=GRID_COLOR)
    axs[0, 1].legend()
    
    # Bottom left: Linearized plot for file 1
    error_mask1 = np.zeros(len(filtered_data1), dtype=bool)
    error_mask1[::5] = True  # Every 5th point
    error_df1 = filtered_data1[error_mask1]
    
    sns.scatterplot(x=filtered_data1['1/T'], y=filtered_data1['ln(R/T^(3/2))'],
                    color=PRIMARY_COLOR, s=40, label='Data Points', ax=axs[1, 0])
    axs[1, 0].errorbar(error_df1['1/T'], error_df1['ln(R/T^(3/2))'],
                      xerr=error_df1['err_x'], yerr=error_df1['err_y_propagated'],
                      fmt='o', color=ERROR_BAR_COLOR, ecolor=ERROR_BAR_EDGE_COLOR, capsize=3,
                      label='Error Bars (every 5th)')
    
    # Plot linear fit for file 1
    x_fit1 = np.linspace(filtered_data1['1/T'].min(), filtered_data1['1/T'].max(), 100)
    y_fit1 = results1['slope'] * x_fit1 + results1['intercept']
    axs[1, 0].plot(x_fit1, y_fit1, '-', color=SECONDARY_COLOR, linewidth=2, label='Linear Fit')
    
    axs[1, 0].set_xlabel('1/T (K^(-1))')
    axs[1, 0].set_ylabel('ln(R/T^(3/2))')
    axs[1, 0].set_title(f"{title1} - Bandgap Analysis ({min_temp_1}K ≤ T ≤ {max_temp_1}K)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, color=GRID_COLOR)
    
    # Add text box with results for file 1
    textstr1 = "\n".join((
        r'$E_g = %.4f \pm %.4f\,eV$' % (results1['bandgap_energy_eV'], results1['bandgap_error_eV']),
        r'$R^2 = %.4f$' % (results1['r_squared']),
        r'$\chi^2_{red} = %.4f$' % (results1['reduced_chi_squared']),
        r'Slope = %.4f' % results1['slope'],
        r'Intercept = %.4f' % results1['intercept'],
        f'Data points: {results1["data_points_used"]}'
    ))
    props = dict(boxstyle='round', facecolor=TEXT_BOX_COLOR, alpha=0.5)
    axs[1, 0].text(0.05, 0.95, textstr1, transform=axs[1, 0].transAxes, fontsize=10,
                  verticalalignment='top', bbox=props)
    
    # Bottom right: Linearized plot for file 2
    error_mask2 = np.zeros(len(filtered_data2), dtype=bool)
    error_mask2[::5] = True  # Every 5th point
    error_df2 = filtered_data2[error_mask2]
    
    sns.scatterplot(x=filtered_data2['1/T'], y=filtered_data2['ln(R/T^(3/2))'],
                    color=PRIMARY_COLOR, s=40, label='Data Points', ax=axs[1, 1])
    axs[1, 1].errorbar(error_df2['1/T'], error_df2['ln(R/T^(3/2))'],
                      xerr=error_df2['err_x'], yerr=error_df2['err_y_propagated'],
                      fmt='o', color=ERROR_BAR_COLOR, ecolor=ERROR_BAR_EDGE_COLOR, capsize=3,
                      label='Error Bars (every 5th)')
    
    # Plot linear fit for file 2
    x_fit2 = np.linspace(filtered_data2['1/T'].min(), filtered_data2['1/T'].max(), 100)
    y_fit2 = results2['slope'] * x_fit2 + results2['intercept']
    axs[1, 1].plot(x_fit2, y_fit2, '-', color=SECONDARY_COLOR, linewidth=2, label='Linear Fit')
    
    axs[1, 1].set_xlabel('1/T (K^(-1))')
    axs[1, 1].set_ylabel('ln(R/T^(3/2))')
    axs[1, 1].set_title(f"{title2} - Bandgap Analysis ({min_temp_2}K ≤ T ≤ {max_temp_2}K)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, color=GRID_COLOR)
    
    # Add text box with results for file 2
    textstr2 = "\n".join((
        r'$E_g = %.4f \pm %.4f\,eV$' % (results2['bandgap_energy_eV'], results2['bandgap_error_eV']),
        r'$R^2 = %.4f$' % (results2['r_squared']),
        r'$\chi^2_{red} = %.4f$' % (results2['reduced_chi_squared']),
        r'Slope = %.4f' % results2['slope'],
        r'Intercept = %.4f' % results2['intercept'],
        f'Data points: {results2["data_points_used"]}'
    ))
    axs[1, 1].text(0.05, 0.95, textstr2, transform=axs[1, 1].transAxes, fontsize=10,
                  verticalalignment='top', bbox=props)
    
    # Add a summary text comparing the two analyses
    # Position it below the plots with adequate spacing
    plt.figtext(0.5, -0.06, 
                f"Comparison of Semiconductor Bandgap Analysis:\n"
                f"{title1}: Eg = {results1['bandgap_energy_eV']:.4f} ± {results1['bandgap_error_eV']:.4f} eV\n"
                f"{title2}: Eg = {results2['bandgap_energy_eV']:.4f} ± {results2['bandgap_error_eV']:.4f} eV", 
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout with additional bottom space for the summary text
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

def load_semiconductor_data(file_path):
    """
    Load semiconductor data from a file containing temperature and resistance measurements.
    """
    try:
        data = pd.read_csv(file_path)
        required_cols = ['Temperature', 'Resistance']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Required columns {required_cols} not found in the file.")
        return data
    except Exception as e:
        print(f"Error reading file with pandas: {e}")
        print("Attempting manual parsing...")
        temperatures, resistances = [], []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if not line.strip() or line.startswith('#') or line.lower().startswith('temperature'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 2:
                temperatures.append(float(parts[0]))
                resistances.append(float(parts[1]))
        return pd.DataFrame({'Temperature': temperatures, 'Resistance': resistances})

# =================== Main Execution ===================

if __name__ == "__main__":
    csv_path_1 = convert_txt_to_csv(DATA_TXT_PATH_1)
    csv_path_2 = convert_txt_to_csv(DATA_TXT_PATH_2)
    
    results1 = analyze_semiconductor_bandgap(csv_path_1, min_temp=MIN_TEMP_1, max_temp=MAX_TEMP_1, custom_title=CUSTOM_TITLE_1)
    results2 = analyze_semiconductor_bandgap(csv_path_2, min_temp=MIN_TEMP_2, max_temp=MAX_TEMP_2, custom_title=CUSTOM_TITLE_2)
    
    fig = plot_2x2_comparison(results1, results2, CUSTOM_TITLE_1, CUSTOM_TITLE_2, MIN_TEMP_1, MAX_TEMP_1, MIN_TEMP_2, MAX_TEMP_2)

    # — NEW: ensure "figures" directory exists and save figure there —
    os.makedirs("figures", exist_ok=True)
    fig.savefig(os.path.join("figures", "bandgap_comparison.png"), dpi=300, bbox_inches='tight')
    print("Saved figure to figures/bandgap_comparison.png")

    # Print summaries (unchanged)
    print(f"\n--- {CUSTOM_TITLE_1} Results ---")
    print(f"Bandgap Energy: {results1['bandgap_energy_eV']:.4f} ± {results1['bandgap_error_eV']:.4f} eV")
    print(f"R-squared: {results1['r_squared']:.4f}")
    print(f"Reduced Chi-squared: {results1['reduced_chi_squared']:.4f}")
    print(f"Number of data points used: {results1['data_points_used']}")

    print(f"\n--- {CUSTOM_TITLE_2} Results ---")
    print(f"Bandgap Energy: {results2['bandgap_energy_eV']:.4f} ± {results2['bandgap_error_eV']:.4f} eV")
    print(f"R-squared: {results2['r_squared']:.4f}")
    print(f"Reduced Chi-squared: {results2['reduced_chi_squared']:.4f}")
    print(f"Number of data points used: {results2['data_points_used']}")

    bandgap_diff = abs(results1['bandgap_energy_eV'] - results2['bandgap_energy_eV'])
    combined_uncertainty = np.sqrt(results1['bandgap_error_eV']**2 + results2['bandgap_error_eV']**2)
    sigma_diff = bandgap_diff / combined_uncertainty

    print(f"\n--- Comparison ---")
    print(f"Bandgap difference: {bandgap_diff:.4f} eV")
    print(f"Combined uncertainty: {combined_uncertainty:.4f} eV")
    print(f"Difference in sigma: {sigma_diff:.2f}σ")

    plt.show()