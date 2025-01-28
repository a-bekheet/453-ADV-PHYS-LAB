import pandas as pd
import matplotlib.pyplot as plt
import sys

def truncate_data_after_zero_bins(data, zero_bin_threshold=250):
    zero_count = 0
    for i, count in enumerate(data['Counts']):
        if count == 0:
            zero_count += 1
        else:
            zero_count = 0
        
        if zero_count >= zero_bin_threshold:
            return data[:i - zero_bin_threshold + 10]
    return data

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
    
    # Truncate data after 20 consecutive zero bins
    data = truncate_data_after_zero_bins(data)
    
    # Plot the data as a scatter plot
    try:
        plt.figure(figsize=(12, 7))
        plt.scatter(data['Time'], data['Counts'], color='blue', edgecolor='k', alpha=0.7)
        plt.title('Muon Lifetime Data', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error while plotting the data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
