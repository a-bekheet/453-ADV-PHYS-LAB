import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Physical constants in SI units
K_BOLTZMANN = 1.380649e-23  # J/K
ELECTRON_CHARGE = 1.602176634e-19  # C
K_BOLTZMANN_EV = 8.617333262e-5  # eV/K (for final result conversion)

def calculate_carrier_mobility(temperature):
    """
    Calculate electron and hole carrier mobilities as a function of temperature
    based on the power law fits shown in the graph.
    
    For electrons: μ_e = 739343109.3882 * T^(-2.3096)
    For holes: μ_h = 177871187.4661 * T^(-2.2500)
    
    Values are converted from cm²/V·s to m²/V·s (SI units)
    
    Args:
        temperature (float or array): Temperature in Kelvin
        
    Returns:
        tuple: (electron_mobility, hole_mobility) in m²/V·s
    """
    # Power law fits from the graph (original in cm²/V·s)
    electron_mobility_cm2 = 739343109.3882 * (temperature ** -2.3096)
    hole_mobility_cm2 = 177871187.4661 * (temperature ** -2.2500)
    
    # Convert from cm²/V·s to m²/V·s (multiply by 1e-4)
    electron_mobility = electron_mobility_cm2 * 1e-4
    hole_mobility = hole_mobility_cm2 * 1e-4
    
    return electron_mobility, hole_mobility

def analyze_semiconductor_bandgap(file_path, min_temp=500, max_temp=None, mobility_correction=True):
    """
    Analyze semiconductor bandgap energy from temperature-resistance data.
    
    This function applies the linearized equation:
    ln(R/T^(3/2)) = (E_g/2k)*(1/T) + ln(A/(μ_e+μ_h))
    
    With mobility correction, it uses:
    ln(R * (μ_e+μ_h) * T^(3/2)) = (E_g/2k)*(1/T) + ln(A)
    
    Args:
        file_path (str): Path to the data file with temperature and resistance values
        min_temp (float): Minimum temperature for the analysis (default: 500K)
        max_temp (float, optional): Maximum temperature for the analysis (default: None)
        mobility_correction (bool): Whether to correct for temperature-dependent mobility (default: True)
        
    Returns:
        dict: Dictionary with analysis results and bandgap energy
    """
    # Load the data with custom parsing
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip the header line and any metadata
    data_lines = []
    for line in lines:
        if line.strip() and not line.startswith('Semiconductor') and not line.startswith('Temperature'):
            data_lines.append(line.strip())
    
    # Parse the data
    temperatures = []
    resistances = []
    for line in data_lines:
        parts = line.split(',')
        if len(parts) == 2:
            try:
                temp = float(parts[0])
                res = float(parts[1])
                temperatures.append(temp)
                resistances.append(res)
            except ValueError:
                print(f"Warning: Could not parse line: {line}")
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Temperature': temperatures,
        'Resistance': resistances
    })
    
    # Filter data for temperatures within the specified range
    filtered_data = data[data['Temperature'] >= min_temp].copy()
    if max_temp is not None:
        filtered_data = filtered_data[filtered_data['Temperature'] <= max_temp].copy()
    
    # Calculate carrier mobilities for each temperature
    electron_mobility, hole_mobility = calculate_carrier_mobility(filtered_data['Temperature'])
    filtered_data['Electron_Mobility'] = electron_mobility
    filtered_data['Hole_Mobility'] = hole_mobility
    filtered_data['Total_Mobility'] = electron_mobility + hole_mobility
    
    # Calculate the parameters for the linearized equation
    filtered_data['1/T'] = 1 / filtered_data['Temperature']
    
    # Apply the appropriate model based on mobility correction option
    if mobility_correction:
        # With mobility correction: ln(R * (μ_e+μ_h) * T^(3/2)) = (E_g/2k)*(1/T) + ln(A)
        # Note: This modification accounts for the temperature dependence of mobilities
        filtered_data['Y'] = np.log(filtered_data['Resistance'] * 
                                    filtered_data['Total_Mobility'] * 
                                    (filtered_data['Temperature'] ** (3/2)))
        y_label = 'ln(R × (μ_e+μ_h) × T^(3/2))'
    else:
        # Original approach: ln(R/T^(3/2)) = (E_g/2k)*(1/T) + ln(A/(μ_e+μ_h))
        filtered_data['Y'] = np.log(filtered_data['Resistance'] / 
                                   (filtered_data['Temperature'] ** (3/2)))
        y_label = 'ln(R/T^(3/2))'
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered_data['1/T'], filtered_data['Y']
    )
    
    # Calculate bandgap energy in Joules first (SI units)
    # The slope equals E_g/2k, so E_g = 2k * slope
    E_g_J = 2 * K_BOLTZMANN * slope  # in J
    E_g_error_J = 2 * K_BOLTZMANN * std_err  # error in J
    
    # Convert to eV for reporting (divide by electron charge)
    E_g = E_g_J / ELECTRON_CHARGE  # in eV
    E_g_error = E_g_error_J / ELECTRON_CHARGE  # error in eV

    # Calculate alternative bandgap value using direct eV formula for verification
    E_g_direct = 2 * K_BOLTZMANN_EV * slope  # in eV directly
    E_g_error_direct = 2 * K_BOLTZMANN_EV * std_err  # error in eV
    
    # Compare the two calculations (should be very close)
    direct_vs_converted_diff = abs(E_g - E_g_direct) / E_g * 100  # percent difference
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original resistance vs temperature data
    ax1.plot(data['Temperature'], data['Resistance'], 'o-', markersize=3, alpha=0.5)
    
    # Label for the analysis region in the legend
    if max_temp is not None:
        analysis_label = f'Analysis Region ({min_temp}K ≤ T ≤ {max_temp}K)'
    else:
        analysis_label = f'Analysis Region (T ≥ {min_temp}K)'
    
    ax1.plot(filtered_data['Temperature'], filtered_data['Resistance'], 'ro', markersize=4,
            label=analysis_label)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title('Semiconductor Resistance vs Temperature')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Carrier mobility vs temperature (log-log scale)
    # Convert back to cm²/V·s for plotting (as this is the conventional unit in the field)
    electron_mobility_cm2 = filtered_data['Electron_Mobility'] * 1e4
    hole_mobility_cm2 = filtered_data['Hole_Mobility'] * 1e4
    
    ax2.loglog(filtered_data['Temperature'], electron_mobility_cm2, 'o-', 
              label='Electrons: μₑ ∝ T^(-2.3096)')
    ax2.loglog(filtered_data['Temperature'], hole_mobility_cm2, 's-', 
              label='Holes: μₕ ∝ T^(-2.2500)')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Carrier Mobility (cm²/V·s)')
    ax2.set_title('Carrier Mobility vs Temperature')
    ax2.grid(True, which='both', ls='--')
    ax2.legend()
    
    # Plot 3: Linearized data and fit
    ax3.scatter(filtered_data['1/T'], filtered_data['Y'], label='Data Points')
    
    # Add the linear fit
    x_fit = np.linspace(filtered_data['1/T'].min(), filtered_data['1/T'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax3.plot(x_fit, y_fit, 'r-', label='Linear Fit')
    
    ax3.set_xlabel('1/T (K^-1)')
    ax3.set_ylabel(y_label)
    
    # Update the plot title to reflect the temperature range and correction method
    correction_method = "with mobility correction" if mobility_correction else "without mobility correction"
    if max_temp is not None:
        ax3.set_title(f'Bandgap Analysis ({min_temp}K ≤ T ≤ {max_temp}K) {correction_method}')
    else:
        ax3.set_title(f'Bandgap Analysis (T ≥ {min_temp}K) {correction_method}')
    
    ax3.legend()
    ax3.grid(True)
    
    # Add text box with results
    textstr = '\n'.join((
        r'$E_g = %.4f \pm %.4f \, \mathrm{eV}$' % (E_g, E_g_error),
        r'$E_g = %.4e \pm %.4e \, \mathrm{J}$' % (E_g_J, E_g_error_J),
        r'$R^2 = %.4f$' % (r_value**2),
        r'Slope = %.4e \, \mathrm{K}$' % (slope),
        r'Intercept = %.4f$' % (intercept),
        f'Data points: {len(filtered_data)}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Results dictionary
    results = {
        'bandgap_energy_J': E_g_J,  # in Joules (SI units)
        'bandgap_error_J': E_g_error_J,  # in Joules
        'bandgap_energy_eV': E_g,  # in electron volts
        'bandgap_error_eV': E_g_error,  # in electron volts
        'bandgap_energy_eV_direct': E_g_direct,  # using K_BOLTZMANN_EV directly
        'bandgap_error_eV_direct': E_g_error_direct,  # using K_BOLTZMANN_EV directly
        'direct_vs_converted_diff_percent': direct_vs_converted_diff,  # difference between methods
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err,
        'data_points_used': len(filtered_data),
        'figure': fig,
        'min_temp_used': min_temp,
        'max_temp_used': max_temp,
        'mobility_correction': mobility_correction
    }
    
    return results

# Main execution
if __name__ == "__main__":
    path = "data_files/feb27_Ge_run1.csv"
    
    # Example 1: Standard analysis without mobility correction
    results_standard = analyze_semiconductor_bandgap(path, min_temp=250, max_temp=350, mobility_correction=False)
    
    print("Analysis without mobility correction:")
    print(f"Bandgap Energy: {results_standard['bandgap_energy_eV']:.4f} ± {results_standard['bandgap_error_eV']:.4f} eV")
    print(f"R-squared: {results_standard['r_squared']:.4f}")
    print(f"Number of data points used: {results_standard['data_points_used']}")
    print(f"Temperature range: {results_standard['min_temp_used']} - {results_standard['max_temp_used']} K")
    
    print("\n" + "-"*50 + "\n")
    
    # Example 2: Analysis with mobility correction
    results_mobility = analyze_semiconductor_bandgap(path, min_temp=250, max_temp=350, mobility_correction=True)
    
    print("Analysis with mobility correction:")
    print(f"Bandgap Energy: {results_mobility['bandgap_energy_eV']:.4f} ± {results_mobility['bandgap_error_eV']:.4f} eV")
    print(f"R-squared: {results_mobility['r_squared']:.4f}")
    print(f"Number of data points used: {results_mobility['data_points_used']}")
    print(f"Temperature range: {results_mobility['min_temp_used']} - {results_mobility['max_temp_used']} K")
    
    plt.show()