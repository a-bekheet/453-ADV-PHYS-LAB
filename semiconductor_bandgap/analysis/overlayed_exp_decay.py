import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from convert_txt_to_csv import convert_txt_to_csv

# Set seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)

# Custom color palette
PALETTE = sns.color_palette("husl", 10)

# Boltzmann constant in eV/K
K_BOLTZMANN = 8.617333262e-5  # eV/K

def analyze_semiconductor_bandgap(file_path, min_temp=500, max_temp=None, custom_title=None, return_data=False):
    """
    Analyze semiconductor bandgap energy from temperature-resistance data.
    
    Args:
        file_path (str): Path to the data file with temperature and resistance values
        min_temp (float): Minimum temperature for the analysis (default: 500K)
        max_temp (float, optional): Maximum temperature for the analysis (default: None)
        custom_title (str, optional): Custom title for the plots (default: None)
        return_data (bool): Whether to return the data for external plotting (default: False)
        
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
    
    # Calculate reduced chi-squared
    y_pred = slope * filtered_data['1/T'] + intercept
    residuals = filtered_data['ln(R/T^(3/2))'] - y_pred
    n = len(filtered_data)
    dof = n - 2  # degrees of freedom (n - number of parameters)
    
    chi_squared = np.sum(residuals**2)
    reduced_chi_squared = chi_squared / dof
    
    # Create the fit data for plotting
    x_fit = np.linspace(filtered_data['1/T'].min(), filtered_data['1/T'].max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Results dictionary
    results = {
        'bandgap_energy_eV': E_g,
        'bandgap_error_eV': E_g_error,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'reduced_chi_squared': reduced_chi_squared,
        'p_value': p_value,
        'std_err': std_err,
        'data_points_used': len(filtered_data),
        'min_temp_used': min_temp,
        'max_temp_used': max_temp,
        'custom_title': custom_title,
        'file_name': file_path
    }
    
    # Add data for external plotting if requested
    if return_data:
        results.update({
            'full_data': data,
            'filtered_data': filtered_data,
            'x_fit': x_fit,
            'y_fit': y_fit
        })
    
    return results


def plot_multiple_analyses(results_list, dpi=120):
    """
    Create polished plots with multiple analysis results using seaborn.
    
    Args:
        results_list (list): List of result dictionaries from analyze_semiconductor_bandgap
        dpi (int): Resolution for the output figures
    
    Returns:
        tuple: Figure objects for the resistance and bandgap plots
    """
    # Create dataframes for plotting
    resistance_data = []
    bandgap_data = []
    fit_lines = []
    
    # Extract sample names for consistent labeling
    sample_names = []
    for i, result in enumerate(results_list):
        file_name = result['file_name'].split('/')[-1].replace('.csv', '').replace('.txt', '')
        if result['custom_title']:
            sample_name = result['custom_title']
        else:
            sample_name = f"Sample {i+1} ({file_name})"
        sample_names.append(sample_name)
    
    # Process each dataset
    for i, result in enumerate(results_list):
        # Resistance data
        df_full = result['full_data'].copy()
        df_full['Sample'] = sample_names[i]
        df_full['DataType'] = 'Full Range'
        
        df_filtered = result['filtered_data'].copy()
        df_filtered['Sample'] = sample_names[i]
        df_filtered['DataType'] = 'Analysis Range'
        
        resistance_data.append(df_full)
        resistance_data.append(df_filtered)
        
        # Bandgap analysis data
        df_bandgap = pd.DataFrame({
            '1/T': result['filtered_data']['1/T'],
            'ln(R/T^(3/2))': result['filtered_data']['ln(R/T^(3/2))'],
            'Sample': sample_names[i]
        })
        bandgap_data.append(df_bandgap)
        
        # Fit lines
        df_fit = pd.DataFrame({
            '1/T': result['x_fit'],
            'ln(R/T^(3/2))': result['y_fit'],
            'Sample': sample_names[i]
        })
        fit_lines.append(df_fit)
    
    # Combine all the data
    resistance_df = pd.concat(resistance_data)
    bandgap_df = pd.concat(bandgap_data)
    fit_df = pd.concat(fit_lines)
    
    # Create results table for display
    results_table = []
    for i, result in enumerate(results_list):
        results_table.append({
            'Sample': sample_names[i],
            'Bandgap (eV)': f"{result['bandgap_energy_eV']:.4f} ± {result['bandgap_error_eV']:.4f}",
            'R²': f"{result['r_squared']:.4f}",
            'χ²ᵣₑₐ': f"{result['reduced_chi_squared']:.4f}",
            'Temp. Range (K)': f"{result['min_temp_used']}−{result['max_temp_used'] or 'max'}",
            'Points': result['data_points_used']
        })
    results_df = pd.DataFrame(results_table)
    
    # ----------- FIGURE 1: RESISTANCE VS TEMPERATURE -----------
    fig_resistance = plt.figure(dpi=dpi)
    
    # Plot full range with lower alpha
    sns.lineplot(
        data=resistance_df[resistance_df['DataType'] == 'Full Range'],
        x='Temperature', y='Resistance',
        hue='Sample', palette=PALETTE[:len(sample_names)],
        alpha=0.3, legend=False, linewidth=1
    )
    
    # Plot analysis range with markers
    sns.scatterplot(
        data=resistance_df[resistance_df['DataType'] == 'Analysis Range'],
        x='Temperature', y='Resistance',
        hue='Sample', palette=PALETTE[:len(sample_names)],
        s=40, alpha=0.8, edgecolor='none'
    )
    
    plt.title('Semiconductor Resistance vs Temperature', fontweight='bold')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Resistance (Ω)')
    plt.legend(title='Sample', bbox_to_anchor=(1.02, 1), loc='upper left')
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    
    # ----------- FIGURE 2: BANDGAP ANALYSIS -----------
    fig_bandgap = plt.figure(dpi=dpi)
    
    # Plot data points
    sns.scatterplot(
        data=bandgap_df, x='1/T', y='ln(R/T^(3/2))',
        hue='Sample', palette=PALETTE[:len(sample_names)],
        s=60, alpha=0.7, edgecolor='none', legend=True
    )
    
    # Plot fit lines
    sns.lineplot(
        data=fit_df, x='1/T', y='ln(R/T^(3/2))',
        hue='Sample', palette=PALETTE[:len(sample_names)],
        linewidth=2, legend=False
    )
    
    plt.title('Semiconductor Bandgap Analysis', fontweight='bold')
    plt.xlabel('1/T (K⁻¹)')
    plt.ylabel('ln(R/T^(3/2))')
    plt.legend(title='Sample', bbox_to_anchor=(1.02, 1), loc='upper left')
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    
    # ----------- FIGURE 3: RESULTS TABLE -----------
    fig_results = plt.figure(figsize=(10, len(results_df)*0.5 + 1), dpi=dpi)
    ax = fig_results.add_subplot(111)
    
    # Hide axes
    ax.axis('off')
    
    # Create a table and modify style
    table = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        loc='center',
        cellLoc='center',
        colColours=[PALETTE[8]]*len(results_df.columns),
        cellColours=[[PALETTE[i % len(PALETTE)], 'white', 'white', 'white', 'white', 'white'] 
                    for i in range(len(results_df))]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add a title
    plt.title('Semiconductor Bandgap Analysis Results', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig_resistance, fig_bandgap, fig_results


# Main execution
if __name__ == "__main__":
    # List of files to analyze
    file_paths = [
        # "data_files/feb27_Ge_run1.txt",
        # "data_files/feb27_Ge_run2.txt",
        # "data_files/feb27_Ge_run3.txt",
        # "data_files/feb27_Ge_run4.txt",
        # "data_files/feb27_Ge_run5.txt",
        # "data_files/feb27_Ge_run6.txt",
        # "data_files/mar4_ptype_test1.txt",
        "data_files/mar4_ptype_test2.txt",
        "data_files/mar4_ptype_test4.txt",
        "data_files/mar4_ptype_test7.txt",
    ]
    
    # Custom titles for each file
    custom_titles = [
        # "Ge Sample 1",
        # "Ge Sample 2",
        # "Ge Sample 3", 
        # "Ge Sample 4",
        # "Ge Sample 5",
        # "Ge Sample 6",
        # "P-Type Test 1",
        "P-Type Test 2",
        "P-Type Test 4",
        "P-Type Test 7",
    ]
    
    # Analysis parameters
    min_temp = 300
    max_temp = 700
    
    # Analyze each file and store results
    results_list = []
    
    for i, file_path in enumerate(file_paths):
        # Convert TXT to CSV if needed
        if file_path.endswith('.txt'):
            path = convert_txt_to_csv(file_path)
        else:
            path = file_path
            
        # Analyze the data
        results = analyze_semiconductor_bandgap(
            path, 
            min_temp=min_temp, 
            max_temp=max_temp, 
            custom_title=custom_titles[i] if i < len(custom_titles) else None,
            return_data=True
        )
        
        # Print results for this file
        print(f"\nResults for {file_path}:")
        print(f"Bandgap Energy: {results['bandgap_energy_eV']:.4f} ± {results['bandgap_error_eV']:.4f} eV")
        print(f"R-squared: {results['r_squared']:.4f}")
        print(f"Reduced Chi-squared: {results['reduced_chi_squared']:.4f}")
        print(f"Number of data points used: {results['data_points_used']}")
        print(f"Temperature range: {results['min_temp_used']} - {results['max_temp_used'] or 'max'} K")
        
        results_list.append(results)
    
    # Create the overlay plots
    fig_resistance, fig_bandgap, fig_results = plot_multiple_analyses(results_list)
    
    # Show the plots
    plt.show()