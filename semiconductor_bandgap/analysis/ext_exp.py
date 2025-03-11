import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from convert_txt_to_csv import convert_txt_to_csv

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Boltzmann constant in eV/K
K_BOLTZMANN = 8.617333262e-5  # eV/K

# Global color variables
PRIMARY_COLOR = '#787878'       # Color for all data points
SECONDARY_COLOR = '#811cea'     # Color for analysis region and fit line
ERROR_BAR_COLOR = '#AA9999'     # Color for points with error bars
ERROR_BAR_EDGE_COLOR = '#AAAAAA'  # Color for error bar edges
ERROR_LINE_COLOR = '#088F8F'    # Color for error lines
RESIDUAL_COLOR = '#691595'      # Color for residuals
CONFIDENCE_BAND_COLOR = '#811cea'  # Color for confidence bands
TEXT_BOX_COLOR = 'lightblue'    # Color for text boxes
GRID_COLOR = '#DDDDDD'          # Color for grid lines

CUSTOM_TITLE = "Silicon (N-Type) Sample"
PATH = "data_files/feb_13_ntype_run1.txt"

def calculate_error_propagation(temperatures, resistances, mean_inv_temp):
    """
    Calculate the uncertainty in bandgap energy using error propagation.
    
    Args:
        temperatures (array): Temperature values in K.
        resistances (array): Resistance values in Ohms.
        mean_inv_temp (float): Mean value of 1/T.
        
    Returns:
        float: The uncertainty in bandgap energy (eV).
        array: The uncertainty in y values (ln(R/T^(3/2))).
    """
    CURRENT = 10e-6  # 10 μA (assumed constant current)
    
    # Compute measured voltage (V = R * I)
    voltages = resistances * CURRENT

    # Voltage and current uncertainties
    delta_V = 0.00015 * voltages + 1.5e-4   # Voltage uncertainty (Volts)
    delta_I = 0.00034 * CURRENT + 200e-9      # Current uncertainty (Amperes)
    
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
    
    Uses the linearized equation:
        ln(R/T^(3/2)) = (E_g/(2k_B))*(1/T) + ln(A/(μ_e+μ_h))
    
    Args:
        file_path (str): Path to the CSV data file.
        min_temp (float): Minimum temperature for analysis.
        max_temp (float, optional): Maximum temperature for analysis.
        custom_title (str, optional): Custom title for the plots.
        
    Returns:
        dict: Analysis results including bandgap energy and fit statistics.
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
    
    # Create the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_palette("colorblind")
    
    # Subplot 1: Resistance vs Temperature
    sns.scatterplot(data=data, x='Temperature', y='Resistance',
                    color=PRIMARY_COLOR, alpha=0.5, s=30, label='All Data', ax=ax1)
    analysis_label = f'Analysis Region (T ≥ {min_temp}K)' if max_temp is None else f'Analysis Region ({min_temp}K ≤ T ≤ {max_temp}K)'
    sns.scatterplot(data=filtered_data, x='Temperature', y='Resistance',
                    color=SECONDARY_COLOR, s=40, label=analysis_label, ax=ax1)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title(custom_title + " - Resistance vs Temperature" if custom_title else "Semiconductor Resistance vs Temperature")
    ax1.grid(True, color=GRID_COLOR)
    ax1.legend()
    
    # Subplot 2: Linearized Data and Fit
    # Plot all data points
    sns.scatterplot(x=filtered_data['1/T'], y=filtered_data['ln(R/T^(3/2))'],
                    color=PRIMARY_COLOR, s=40, label='Data Points', ax=ax2)
    # Plot error bars on every 5th point
    error_mask = np.zeros(len(filtered_data), dtype=bool)
    error_mask[::5] = True
    error_df = filtered_data[error_mask]
    ax2.errorbar(error_df['1/T'], error_df['ln(R/T^(3/2))'],
                 xerr=error_df['err_x'], yerr=error_df['err_y_propagated'],
                 fmt='o', color=ERROR_BAR_COLOR, ecolor=ERROR_BAR_EDGE_COLOR, capsize=3,
                 label='Error Bars (every 5th)')
    # Plot linear fit
    x_fit = np.linspace(filtered_data['1/T'].min(), filtered_data['1/T'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax2.plot(x_fit, y_fit, '-', color=SECONDARY_COLOR, linewidth=2, label='Linear Fit')
    
    # 95% confidence band (approximate)
    t_val = stats.t.ppf(0.975, dof)
    x_mean = np.mean(filtered_data['1/T'])
    s_err = np.sqrt(np.sum((filtered_data['ln(R/T^(3/2))'] - y_pred)**2) / dof)
    s_x = np.sqrt(np.sum((filtered_data['1/T'] - x_mean)**2))
    y_err = t_val * s_err * np.sqrt(1/n + (x_fit - x_mean)**2 / s_x**2)
    ax2.fill_between(x_fit, y_fit - y_err, y_fit + y_err, alpha=0.2, color=CONFIDENCE_BAND_COLOR,
                     label='95% Confidence Band')
    ax2.set_xlabel('1/T (K^(-1))')
    ax2.set_ylabel('ln(R/T^(3/2))')
    title2 = (f"{custom_title} - Bandgap Analysis ({min_temp}K ≤ T ≤ {max_temp}K)" 
              if max_temp else f"{custom_title} - Bandgap Analysis (T ≥ {min_temp}K)")
    ax2.set_title(title2 if custom_title else ("Semiconductor Bandgap Analysis" + (f" ({min_temp}K ≤ T ≤ {max_temp}K)" if max_temp else " (T ≥ {min_temp}K)")))
    ax2.legend()
    ax2.grid(True, color=GRID_COLOR)
    
    # Add a text box with key results on ax2
    textstr = "\n".join((
        r'$E_g = %.4f \pm %.4f\,eV$' % (E_g, E_g_error),
        r'$R^2 = %.4f$' % (r_value**2),
        r'$\chi^2_{red} = %.4f$' % (reduced_chi_squared),
        r'Slope = %.4f' % slope,
        r'Intercept = %.4f' % intercept,
        f'Data points: {n}'
    ))
    props = dict(boxstyle='round', facecolor=TEXT_BOX_COLOR, alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
# --- Improved Subplot 3: Error Propagation & Residuals ---

# Plot propagated error vs Temperature on ax3 (primary y-axis)
    ax3.plot(filtered_data['Temperature'], filtered_data['err_y_propagated'], 'o-', 
            color=ERROR_LINE_COLOR, alpha=0.6, markersize=3, label='Propagated Error in y')
    # Highlight every 5th point
    ax3.plot(filtered_data.loc[error_mask, 'Temperature'], 
            filtered_data.loc[error_mask, 'err_y_propagated'], 'o', 
            color=ERROR_BAR_COLOR, markersize=6, label='Selected Error Points')

    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('Error in ln(R/T^(3/2))', color=ERROR_LINE_COLOR)
    ax3.tick_params(axis='y', labelcolor=ERROR_LINE_COLOR)
    ax3.grid(True, color=GRID_COLOR)

    # Calculate error contributions
    CURRENT = 10e-6
    voltages = filtered_data['Resistance'] * CURRENT
    delta_V = 0.00015 * voltages + 1.5e-3
    delta_I = 0.00034 * CURRENT + 200e-9
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

    # Print mean contributions to console
    print("\nMean Error Contributions:")
    print(f"Voltage: {filtered_data['V_contrib_pct'].mean():.2f}%")
    print(f"Current: {filtered_data['I_contrib_pct'].mean():.2f}%")
    print(f"Temperature: {filtered_data['T_contrib_pct'].mean():.2f}%")

    # Plot fit residuals on a twin axis (secondary y-axis)
    ax3_twin = ax3.twinx()
    ax3_twin.scatter(filtered_data['Temperature'], residuals, 
                    color=RESIDUAL_COLOR, alpha=0.5, s=30, label='Fit Residuals')
    ax3_twin.plot(filtered_data['Temperature'], residuals, '--', 
                color=RESIDUAL_COLOR, alpha=0.5)
    ax3_twin.set_ylabel('Fit Residuals', color=RESIDUAL_COLOR)
    ax3_twin.tick_params(axis='y', labelcolor=RESIDUAL_COLOR)
    ax3_twin.axhline(0, color=RESIDUAL_COLOR, linestyle='--', alpha=0.3)

    # Add horizontal lines for ±1σ residuals
    std_res = np.std(residuals)
    ax3_twin.axhline(std_res, color=RESIDUAL_COLOR, linestyle=':', alpha=0.5, label=f'±1σ (±{std_res:.4f})')
    ax3_twin.axhline(-std_res, color=RESIDUAL_COLOR, linestyle=':', alpha=0.5)

    # Merge legends from both axes
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    # Add a text box with error breakdown
    textstr3 = "\n".join((
        "Error Contributions:",
        f"Voltage: {filtered_data['V_contrib_pct'].mean():.1f}%",
        f"Current: {filtered_data['I_contrib_pct'].mean():.1f}%",
        f"Temperature: {filtered_data['T_contrib_pct'].mean():.1f}%"
    ))
    props3 = dict(boxstyle='round', facecolor=TEXT_BOX_COLOR, alpha=0.5)
    ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', bbox=props3)

    ax3.set_title('Error Propagation & Residuals')

    plt.tight_layout()
    plt.show()

    
    # --- Weighted Linear Fit ---
    # Use np.polyfit with weights = 1/err_y_propagated
    weights = 1.0 / (filtered_data['err_y_propagated']**2)
    # Use sqrt(weights) because polyfit expects weights such that the residuals are scaled
    p, cov = np.polyfit(filtered_data['1/T'], filtered_data['ln(R/T^(3/2))'], deg=1, w=np.sqrt(weights), cov=True)
    weighted_slope = p[0]
    weighted_intercept = p[1]
    weighted_E_g = 2 * K_BOLTZMANN * weighted_slope
    weighted_E_g_err = 2 * K_BOLTZMANN * np.sqrt(cov[0, 0])
    # Compute weighted R²
    y_weighted_pred = weighted_slope * filtered_data['1/T'] + weighted_intercept
    weighted_r_squared = 1 - np.sum((filtered_data['ln(R/T^(3/2))'] - y_weighted_pred)**2) / np.sum((filtered_data['ln(R/T^(3/2))'] - np.mean(filtered_data['ln(R/T^(3/2))']))**2)
    
    # Add weighted fit results to a results dictionary
    results = {
        'bandgap_energy_eV': E_g,
        'bandgap_error_eV': E_g_error,
        'bandgap_std_err_eV': E_g_std_err,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'reduced_chi_squared': reduced_chi_squared,
        'p_value': p_value,
        'std_err': std_err,
        'data_points_used': n,
        'min_temp_used': min_temp,
        'max_temp_used': max_temp,
        'custom_title': custom_title,
        'weighted_bandgap_energy_eV': weighted_E_g,
        'weighted_bandgap_error_eV': weighted_E_g_err,
        'weighted_r_squared': weighted_r_squared,
        'weighted_slope': weighted_slope,
        'weighted_intercept': weighted_intercept,
        'err_y_propagated': filtered_data['err_y_propagated'].values,
        'residuals': residuals.values,
        'figure': fig
    }
    
    return results

def load_semiconductor_data(file_path):
    """
    Load semiconductor data from a file containing temperature and resistance measurements.
    Expected format: Temperature,Resistance
    
    Args:
        file_path (str): Path to the data file.
        
    Returns:
        pd.DataFrame: DataFrame with the loaded data.
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

# --- Main Execution ---
if __name__ == "__main__":
    # Choose your input file (conversion function assumed to work)
    txtpath = "data_files/feb_13_ntype_run1.txt"
    csv_path = convert_txt_to_csv(txtpath)
    
    results = analyze_semiconductor_bandgap(csv_path, min_temp=520, max_temp=600, custom_title=CUSTOM_TITLE)
    
    print(f"Bandgap Energy: {results['bandgap_energy_eV']:.4f} ± {results['bandgap_error_eV']:.4f} eV")
    print(f"Standard Error from regression: {results['bandgap_std_err_eV']:.4f} eV")
    print(f"R-squared: {results['r_squared']:.4f}")
    print(f"Reduced Chi-squared: {results['reduced_chi_squared']:.4f}")
    print(f"Number of data points used: {results['data_points_used']}")
    print(f"Temperature range: {results['min_temp_used']} - {results['max_temp_used']} K")
    
    print("\nError Analysis:")
    print(f"Mean propagated error in y: {np.mean(results['err_y_propagated']):.6f}")
    print(f"Standard deviation of fit residuals: {np.std(results['residuals']):.6f}")
    print(f"Error in bandgap from propagation: {results['bandgap_error_eV']:.6f} eV")
    print(f"Error in bandgap from fit std error: {results['bandgap_std_err_eV']:.6f} eV")
    
    print("\nWeighted Fit Results (using 1/σ² weights):")
    print(f"Weighted bandgap: {results['weighted_bandgap_energy_eV']:.6f} ± {results['weighted_bandgap_error_eV']:.6f} eV")
    print(f"Weighted R-squared: {results['weighted_r_squared']:.6f}")
    
    plt.show()
