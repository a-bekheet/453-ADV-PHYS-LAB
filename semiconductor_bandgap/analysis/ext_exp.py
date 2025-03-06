import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from convert_txt_to_csv import convert_txt_to_csv

# Boltzmann constant in eV/K
K_BOLTZMANN = 8.617333262e-5  # eV/K

# Global color variables
PRIMARY_COLOR = '#787878'    # Color for all data points
SECONDARY_COLOR = '#811cea'   # Color for analysis region and fit line

def analyze_semiconductor_bandgap(file_path, min_temp=500, max_temp=None):
    """
    Analyze semiconductor bandgap energy from temperature-resistance data.
    
    This function applies the linearized equation:
    ln(R/T^(3/2)) = (E_g/2k)*(1/T) + ln(A/(μ_e+μ_h))
    
    Args:
        file_path (str): Path to the data file with temperature and resistance values
        min_temp (float): Minimum temperature for the analysis (default: 500K)
        max_temp (float, optional): Maximum temperature for the analysis (default: None)
        
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
    
    # Calculate the parameters for the linearized equation
    filtered_data['1/T'] = 1 / filtered_data['Temperature']
    filtered_data['ln(R/T^(3/2))'] = np.log(filtered_data['Resistance'] / (filtered_data['Temperature'] ** (3/2)))
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered_data['1/T'], filtered_data['ln(R/T^(3/2))']
    )
    
    # Calculate bandgap energy
    E_g = 2 * K_BOLTZMANN * slope  # in eV
    E_g_error = 2 * K_BOLTZMANN * std_err  # error in eV
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Original resistance vs temperature data
    ax1.plot(data['Temperature'], data['Resistance'], 'o-', color=PRIMARY_COLOR, markersize=3, alpha=0.5)
    
    # Label for the analysis region in the legend
    if max_temp is not None:
        analysis_label = f'Analysis Region ({min_temp}K ≤ T ≤ {max_temp}K)'
    else:
        analysis_label = f'Analysis Region (T ≥ {min_temp}K)'
    
    ax1.plot(filtered_data['Temperature'], filtered_data['Resistance'], 'o', 
             color=SECONDARY_COLOR, markersize=4, label=analysis_label)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title('Semiconductor Resistance vs Temperature')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Linearized data and fit
    ax2.scatter(filtered_data['1/T'], filtered_data['ln(R/T^(3/2))'], 
                color=PRIMARY_COLOR, label='Data Points')
    
    # Add the linear fit
    x_fit = np.linspace(filtered_data['1/T'].min(), filtered_data['1/T'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax2.plot(x_fit, y_fit, '-', color=SECONDARY_COLOR, label='Linear Fit')
    
    ax2.set_xlabel('1/T (K^-1)')
    ax2.set_ylabel('ln(R/T^(3/2))')
    
    # Update the plot title to reflect the temperature range
    if max_temp is not None:
        ax2.set_title(f'Semiconductor Bandgap Analysis ({min_temp}K ≤ T ≤ {max_temp}K)')
    else:
        ax2.set_title(f'Semiconductor Bandgap Analysis (T ≥ {min_temp}K)')
    
    ax2.legend()
    ax2.grid(True)
    
    # Add text box with results
    textstr = '\n'.join((
        r'$E_g = %.4f \pm %.4f \, \mathrm{eV}$' % (E_g, E_g_error),
        r'$R^2 = %.4f$' % (r_value**2),
        r'Slope = %.4f$' % (slope),
        r'Intercept = %.4f$' % (intercept),
        f'Data points: {len(filtered_data)}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Results dictionary
    results = {
        'bandgap_energy_eV': E_g,
        'bandgap_error_eV': E_g_error,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err,
        'data_points_used': len(filtered_data),
        'figure': fig,
        'min_temp_used': min_temp,
        'max_temp_used': max_temp
    }
    
    return results

# Main execution
if __name__ == "__main__":
    # txtpath = ["feb_13_RoomTempHold.txt", "feb27_Ge_run1.txt", "feb27_Ge_run2.txt", "feb27_Ge_run3.txt"]
    txtpath = "data_files/feb27_Ge_run1.txt"
    path = convert_txt_to_csv(txtpath)
    # path = "data_files/mar4_ptype_test1.csv"
    
    # Example usage with both min and max temperature specified
    results = analyze_semiconductor_bandgap(path, min_temp=320, max_temp=400)
    
    print(f"Bandgap Energy: {results['bandgap_energy_eV']:.4f} ± {results['bandgap_error_eV']:.4f} eV")
    print(f"R-squared: {results['r_squared']:.4f}")
    print(f"Number of data points used: {results['data_points_used']}")
    print(f"Temperature range: {results['min_temp_used']} - {results['max_temp_used']} K")
    
    plt.show()