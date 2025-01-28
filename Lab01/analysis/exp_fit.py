import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
import numpy as np

def truncate_data_after_zero_bins(data, zero_bin_threshold=250):
    """
    Truncate the data after encountering a specified number of consecutive zero counts.

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
            print(f"Truncating data at index {truncate_index} (Time = {data['Time'].iloc[truncate_index]:.2f} s)")
            break

    return data[:truncate_index]

def exponential_decay(t, A, tau, C):
    """
    Exponential decay function.

    Parameters:
    - t (float or np.array): Time variable.
    - A (float): Amplitude.
    - tau (float): Decay constant (lifetime).
    - C (float): Offset (background count).

    Returns:
    - float or np.array: The value of the exponential decay at time t.
    """
    return A * np.exp(-t / tau) + C

def main():
    # Path to the CSV file
    csv_file_path = 'Lab01/analysis/jan28data_muonlifetime.csv'

    # List of possible separators to try
    separators = [',', '\t', ';']
    data = None

    print("Attempting to read the CSV file with different separators...")

    for sep in separators:
        try:
            # Attempt to read the CSV with the current separator
            data = pd.read_csv(csv_file_path, sep=sep, header=None, names=['Time', 'Counts'])

            # Check if 'Counts' column has non-NaN values
            if data['Counts'].notna().sum() > 0:
                print(f"Successfully read the CSV file using separator: '{sep}'")
                break  # Exit the loop if successful
        except Exception as e:
            print(f"Failed to read with separator '{sep}': {e}")
            continue  # Try the next separator

    # If none of the separators worked, try inferring the separator
    if data is None or data['Counts'].isna().all():
        try:
            print("Attempting to infer the separator...")
            data = pd.read_csv(csv_file_path, sep=None, engine='python', header=None, names=['Time', 'Counts'])
            print("Successfully read the CSV file by inferring the separator.")
        except Exception as e:
            print(f"Failed to read the CSV file with inferred separator: {e}")
            sys.exit(1)

    # Verify the data
    print("\nData Preview:")
    print(data.head())

    # Check for any NaN values in 'Counts' column
    if data['Counts'].isnull().any():
        print("\nWarning: Some 'Counts' values are NaN. There might be an issue with the separator or data formatting.")
        print("Please verify the CSV file's structure.")
        sys.exit(1)

    # Convert 'Time' and 'Counts' to numeric types, if they aren't already
    try:
        data['Time'] = pd.to_numeric(data['Time'])
        data['Counts'] = pd.to_numeric(data['Counts'])
    except ValueError as e:
        print(f"Error converting data to numeric types: {e}")
        sys.exit(1)

    # Truncate data after 250 consecutive zero counts to remove noisy data
    data = truncate_data_after_zero_bins(data, zero_bin_threshold=250)

    # Remove any remaining zero or negative counts to improve fitting
    data = data[data['Counts'] > 0]

    # Prepare data for fitting
    t_data = data['Time'].values
    counts_data = data['Counts'].values

    # Initial guess for the parameters A, tau, and C
    initial_guess = [max(counts_data), 1.0, min(counts_data)]

    try:
        # Perform the curve fit
        popt, pcov = curve_fit(exponential_decay, t_data, counts_data, p0=initial_guess)

        # Extract the parameters and their uncertainties
        A_fit, tau_fit, C_fit = popt
        A_err, tau_err, C_err = np.sqrt(np.diag(pcov))

        print("\nFitting Results:")
        print(f"A (Amplitude): {A_fit:.2f} ± {A_err:.2f}")
        print(f"tau (Lifetime): {tau_fit:.2f} ± {tau_err:.2f} s")
        print(f"C (Background): {C_fit:.2f} ± {C_err:.2f}")
    except Exception as e:
        print(f"Error during curve fitting: {e}")
        sys.exit(1)

    # Generate fitted curve data
    t_fit = np.linspace(min(t_data), max(t_data), 500)
    counts_fit = exponential_decay(t_fit, *popt)

    # Calculate residuals
    residuals = counts_data - exponential_decay(t_data, *popt)

    # Plot the data and the fit
    try:
        plt.figure(figsize=(14, 8))

        # Plot the experimental data
        plt.scatter(t_data, counts_data, color='blue', edgecolor='k', alpha=0.7, label='Experimental Data')

        # Plot the fitted exponential curve
        plt.plot(t_fit, counts_fit, color='red', linewidth=2, label='Exponential Fit')

        # Add labels and title
        plt.title('Muon Lifetime Data with Exponential Fit', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        # Display the fitted parameters on the plot
        textstr = '\n'.join((
            r'$A=%.2f \pm %.2f$' % (A_fit, A_err),
            r'$\tau=%.2f \pm %.2f$ s' % (tau_fit, tau_err),
            r'$C=%.2f \pm %.2f$' % (C_fit, C_err)))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error while plotting the data and fit: {e}")
        sys.exit(1)

    # Plot residuals
    try:
        plt.figure(figsize=(14, 4))
        plt.scatter(t_data, residuals, color='green', edgecolor='k', alpha=0.6)
        plt.hlines(0, min(t_data), max(t_data), colors='red', linestyles='dashed')
        plt.title('Residuals of the Fit', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Residuals (Counts)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error while plotting residuals: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
