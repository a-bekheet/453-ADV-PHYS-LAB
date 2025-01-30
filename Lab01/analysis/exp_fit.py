import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
import numpy as np
import os

def read_csv_file(csv_file_path, separators=[',', '\t', ';']):
    """
    Attempts to read a CSV file using a list of possible separators.
    If all separators fail, it tries to infer the separator.

    Parameters:
    - csv_file_path (str): Path to the CSV file.
    - separators (list): List of separators to try.

    Returns:
    - pd.DataFrame: DataFrame with 'Time' and 'Counts' columns.
    """
    data = None
    print("Attempting to read the CSV file with different separators...")
    for sep in separators:
        try:
            data = pd.read_csv(csv_file_path, sep=sep, header=None, names=['Time', 'Counts'])
            if data['Counts'].notna().sum() > 0:
                print(f"Successfully read the CSV file using separator: '{sep}'")
                return data
        except Exception as e:
            print(f"Failed to read with separator '{sep}': {e}")
            continue

    # If none of the separators worked, try inferring the separator
    try:
        print("Attempting to infer the separator...")
        data = pd.read_csv(csv_file_path, sep=None, engine='python', header=None, names=['Time', 'Counts'])
        print("Successfully read the CSV file by inferring the separator.")
        return data
    except Exception as e:
        print(f"Failed to read the CSV file with inferred separator: {e}")
        sys.exit(1)

def truncate_data_after_zero_bins(data, zero_bin_threshold=250):
    """
    Truncate the data after encountering a specified number 
    of consecutive zero counts.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing 'Counts' column.
    - zero_bin_threshold (int): The number of consecutive zeros to trigger truncation.

    Returns:
    - pd.DataFrame: The truncated DataFrame.
    """
    zero_count = 0
    truncate_index = len(data)  # Default to not truncating

    for i, count in enumerate(data['Counts']):
        if count == 0:
            zero_count += 1
        else:
            zero_count = 0

        if zero_count >= zero_bin_threshold:
            truncate_index = i - zero_bin_threshold + 10  # Keep a few points before truncation
            print(f"Truncating data at index {truncate_index} (Time = {data['Time'].iloc[truncate_index]:.2f} µs)")
            break

    return data[:truncate_index]

def exponential_decay(t, A, tau, C):
    """
    Exponential decay function.

    Parameters:
    - t (float or np.array): Time (in microseconds).
    - A (float): Amplitude.
    - tau (float): Decay constant (lifetime) in microseconds.
    - C (float): Offset (background count).

    Returns:
    - float or np.array: The exponential decay value at t.
    """
    return A * np.exp(-t / tau) + C

def filter_time_bounds(data, lower_bound, upper_bound):
    """
    Filters the data based on lower and upper time (Voltage) bounds.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing 'Time' and 'Counts'.
    - lower_bound (float): Lower bound for Time (inclusive).
    - upper_bound (float): Upper bound for Time (inclusive).

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    filtered_data = data[(data['Time'] >= lower_bound) & (data['Time'] <= upper_bound)]
    print(f"Data filtered to {lower_bound} V ≤ Time ≤ {upper_bound} V. Number of points: {len(filtered_data)}")
    return filtered_data

def convert_time_units(data, conversion_factor=2.0):
    """
    Converts Time from Volts to microseconds.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing 'Time' and 'Counts'.
    - conversion_factor (float): Factor to convert Time.

    Returns:
    - pd.DataFrame: DataFrame with converted Time.
    """
    data = data.copy()
    data['Time'] = data['Time'] * conversion_factor  # Now in microseconds
    print(f"Time converted from Volts to microseconds using factor {conversion_factor}.")
    return data

def perform_curve_fit(t_data, counts_data, initial_guess=None):
    """
    Performs curve fitting using the exponential decay model.

    Parameters:
    - t_data (np.array): Time data in microseconds.
    - counts_data (np.array): Counts data.
    - initial_guess (list or tuple): Initial guess for [A, tau, C].

    Returns:
    - tuple: Optimal parameters and their uncertainties.
    """
    try:
        # Convert input data to numpy arrays
        t_data = np.array(t_data, dtype=float)
        counts_data = np.array(counts_data, dtype=float)
        
        # Set initial guess if not provided
        if initial_guess is None:
            initial_guess = [np.max(counts_data), 1.0, np.min(counts_data)]
        
        # Perform the curve fit
        popt, pcov = curve_fit(exponential_decay, t_data, counts_data, p0=initial_guess)
        
        # Calculate uncertainties
        perr = np.sqrt(np.diag(pcov))
        
        return popt, perr
    except Exception as e:
        print(f"Error in curve fitting: {str(e)}")
        raise

def plot_fit(t_data, counts_data, popt, perr, bounds, save_fig=False, fig_name_prefix='muon_lifetime'):
    """
    Plots the experimental data, the exponential fit, and residuals.

    Parameters:
    - t_data (np.array): Time data in microseconds.
    - counts_data (np.array): Counts data.
    - popt (tuple): Optimal fit parameters (A, tau, C).
    - perr (tuple): Uncertainties in fit parameters (A_err, tau_err, C_err).
    - bounds (tuple): (lower_bound, upper_bound) for the current iteration.
    - save_fig (bool): Whether to save the figures as files.
    - fig_name_prefix (str): Prefix for the saved figure filenames.
    """
    A_fit, tau_fit, C_fit = popt
    A_err, tau_err, C_err = perr
    t_fit = np.linspace(min(t_data), max(t_data), 500)
    counts_fit = exponential_decay(t_fit, *popt)
    residuals = counts_data - exponential_decay(t_data, *popt)

    # Plot the data and the fit
    plt.figure(figsize=(14, 8))
    plt.scatter(t_data, counts_data, color='blue', edgecolor='k', alpha=0.7, label='Experimental Data')
    plt.plot(t_fit, counts_fit, color='red', linewidth=2, label='Exponential Fit')
    plt.title(f'Muon Lifetime Data {bounds[0]}V - {bounds[1]}V', fontsize=16)
    plt.xlabel('Time (µs)', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Display the fitted parameters
    textstr = '\n'.join((
        r'$A=%.2f \pm %.2f$' % (float(A_fit), float(A_err)),
        r'$\tau=%.2f \pm %.2f$ µs' % (float(tau_fit), float(tau_err)),
        r'$C=%.2f \pm %.2f$' % (float(C_fit), float(C_err))))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=props)

    if save_fig:
        fig_filename = f"{fig_name_prefix}_{bounds[0]}V_{bounds[1]}V_fit.png"
        plt.savefig(fig_filename)
        print(f"Fit plot saved as {fig_filename}")
    
    plt.tight_layout()
    plt.show()

    # Plot residuals
    plt.figure(figsize=(14, 4))
    plt.scatter(t_data, residuals, color='green', edgecolor='k', alpha=0.6)
    plt.hlines(0, min(t_data), max(t_data), colors='red', linestyles='dashed')
    plt.title('Residuals of the Fit', fontsize=16)
    plt.xlabel('Time (µs)', fontsize=14)
    plt.ylabel('Residuals (Counts)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_fig:
        residuals_filename = f"{fig_name_prefix}_{bounds[0]}V_{bounds[1]}V_residuals.png"
        plt.savefig(residuals_filename)
        print(f"Residuals plot saved as {residuals_filename}")
    plt.tight_layout()
    plt.show()

def process_bounds(data, bounds_list, zero_bin_threshold=250, save_fig=False):
    """
    Processes the data for each set of bounds: filtering, fitting, and plotting.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing 'Time' and 'Counts'.
    - bounds_list (list of tuples): List of (lower_bound, upper_bound) in Volts.
    - zero_bin_threshold (int): Threshold for truncating data after consecutive zeros.
    - save_fig (bool): Whether to save the figures as files.

    Returns:
    - list of dicts: Fitting results for each bounds.
    """
    results = []
    for bounds in bounds_list:
        print(f"\nProcessing bounds: {bounds[0]}V - {bounds[1]}V")

        # Filter data based on bounds
        filtered_data = filter_time_bounds(data, bounds[0], bounds[1])

        if filtered_data.empty:
            print(f"No data in the range {bounds[0]}V - {bounds[1]}V. Skipping...")
            continue

        # Convert Time units
        converted_data = convert_time_units(filtered_data)

        # Truncate data after consecutive zeros
        truncated_data = truncate_data_after_zero_bins(converted_data, zero_bin_threshold)

        # Remove zero or negative counts
        cleaned_data = truncated_data[truncated_data['Counts'] > 0]

        if cleaned_data.empty:
            print("No positive counts after cleaning. Skipping...")
            continue

        # Prepare data for fitting
        t_data = cleaned_data['Time'].values
        counts_data = cleaned_data['Counts'].values

        # Perform curve fitting
        try:
            popt, perr = perform_curve_fit(t_data, counts_data)
            A_fit, tau_fit, C_fit = popt
            A_err, tau_err, C_err = perr
            
            # Convert to Python floats for printing
            A_fit_float = float(A_fit)
            tau_fit_float = float(tau_fit)
            C_fit_float = float(C_fit)
            A_err_float = float(A_err)
            tau_err_float = float(tau_err)
            C_err_float = float(C_err)
            
            print("Fitting Results:")
            print(f"A (Amplitude): {A_fit_float:.2f} ± {A_err_float:.2f}")
            print(f"tau (Lifetime, µs): {tau_fit_float:.2f} ± {tau_err_float:.2f}")
            print(f"C (Background): {C_fit_float:.2f} ± {C_err_float:.2f}")
        except Exception as e:
            print(f"Error during curve fitting: {e}")
            continue

        # Store results
        result = {
            'bounds': bounds,
            'A_fit': float(A_fit),
            'A_err': float(A_err),
            'tau_fit': float(tau_fit),
            'tau_err': float(tau_err),
            'C_fit': float(C_fit),
            'C_err': float(C_err)
        }
        results.append(result)

        # Plot the fit and residuals
        plot_fit(t_data, counts_data, popt, perr, bounds, save_fig)

    return results

def main():
    # Path to the CSV file
    csv_file_path = 'Lab01/analysis/jan30data_muonlifetime.csv'  # Update this path as needed

    # Verify if the file exists
    if not os.path.isfile(csv_file_path):
        print(f"CSV file not found at path: {csv_file_path}")
        sys.exit(1)

    # Read the data
    data = read_csv_file(csv_file_path)

    # Data preview
    print("\nData Preview:")
    print(data.head())

    # Check for any NaN values in 'Counts' column
    if data['Counts'].isnull().any():
        print("\nWarning: Some 'Counts' values are NaN. Verify the CSV file's structure.")
        sys.exit(1)

    # Convert to numeric
    try:
        data['Time'] = pd.to_numeric(data['Time'])
        data['Counts'] = pd.to_numeric(data['Counts'])
    except ValueError as e:
        print(f"Error converting data to numeric: {e}")
        sys.exit(1)

    # Define list of bounds to iterate through (in Volts)
    bounds_list = [
        (0, 3.5),
        (0, 4.0),
        (0, 4.5),
        (0, 5),
        (0, 5.5),
        (0, 6.0)
    ]

    # Process each set of bounds
    fitting_results = process_bounds(data, bounds_list, zero_bin_threshold=250, save_fig=False)

    # Print summary of fitting results
    print("\nSummary of Fitting Results:")
    for res in fitting_results:
        lb, ub = res['bounds']
        print(f"Bounds: {lb}V - {ub}V")
        print(f"  A = {res['A_fit']:.2f} ± {res['A_err']:.2f}")
        print(f"  tau = {res['tau_fit']:.2f} ± {res['tau_err']:.2f} µs")
        print(f"  C = {res['C_fit']:.2f} ± {res['C_err']:.2f}\n")

if __name__ == "__main__":
    main()