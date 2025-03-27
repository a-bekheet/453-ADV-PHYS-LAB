import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.signal import find_peaks

def extract_angle_range(filename):
    """Extract angle range from the CSV file header."""
    # Open file with cp1252 encoding (Windows-1252)
    with open(filename, 'r', encoding='cp1252') as f:
        header = f.readline().strip()
    
    # Extract the from-to angle information
    angle_info = header.split(',')[-1].strip()
    # Use a more flexible pattern to match the degree symbol in different encodings
    match = re.search(r'from\s+(\d+).*to\s+(\d+).*at\s+(\d+)', angle_info)
    
    if match:
        start_angle = float(match.group(1))
        end_angle = float(match.group(2))
        return start_angle, end_angle
    else:
        # Default range if pattern not found
        print(f"Warning: Could not extract angle range from {filename}. Using default range.")
        return 100.0, 30.0

def process_xray_file(file_path):
    """Process a single X-ray CSV file and return dataframe with angle and intensity."""
    print(f"Processing {file_path}...")
    
    # Read the CSV file with correct encoding
    df = pd.read_csv(file_path, header=0, encoding='cp1252')
    
    # Extract the intensity values from the first column
    intensity_col = df.columns[0]  # Usually "X-ray scan"
    intensity_values = df[intensity_col].values
    
    # Extract angle range from file header
    start_angle, end_angle = extract_angle_range(file_path)
    
    # Create angle array
    # Note: We're going from high angle to low angle based on the header information
    angles = np.linspace(start_angle, end_angle, len(intensity_values))
    
    # Create a new dataframe with angle and intensity
    result_df = pd.DataFrame({
        'Angle (degrees)': angles,
        'Intensity (a.u.)': intensity_values
    })
    
    # Get sample name from filename (remove path and extension)
    sample_name = os.path.basename(file_path).split('.')[0]
    
    return result_df, sample_name

def find_significant_peaks(df, prominence_factor=0.1, width_threshold=10, min_peak_distance=1.0):
    """Find significant peaks in the XRD data."""
    # Get the intensity and angle data
    intensity = df['Intensity (a.u.)'].values
    angles = df['Angle (degrees)'].values
    
    # Calculate the prominence threshold as a fraction of the max intensity range
    intensity_range = np.max(intensity) - np.min(intensity)
    prominence_threshold = prominence_factor * intensity_range
    
    # Find peaks with sufficient prominence
    peaks, properties = find_peaks(intensity, prominence=prominence_threshold, width=1)
    
    # Filter peaks by width if there are too many
    if len(peaks) > 20:
        # Filter out peaks that are too wide
        peak_widths = properties['width']
        peaks = peaks[peak_widths < width_threshold]
    
    # Create a list of (peak_index, angle, intensity, prominence) for each peak
    raw_peak_data = [(i, angles[i], intensity[i], properties['prominences'][j]) 
                     for j, i in enumerate(peaks) if j < len(properties['prominences'])]
    
    # Sort by prominence (highest first)
    sorted_peaks = sorted(raw_peak_data, key=lambda x: x[3], reverse=True)
    
    # Filter closely spaced peaks (keep only the first one within min_peak_distance)
    filtered_peaks = []
    processed_angles = set()
    
    for idx, angle, intensity, prominence in sorted_peaks:
        # Check if this peak is close to any already processed peak
        is_close_to_existing = False
        for existing_angle in processed_angles:
            if abs(angle - existing_angle) < min_peak_distance:
                is_close_to_existing = True
                break
                
        # If not close to any existing peak, add it
        if not is_close_to_existing:
            filtered_peaks.append((angle, intensity))
            processed_angles.add(angle)
            
            # Limit to top 10 peaks
            if len(filtered_peaks) >= 10:
                break
    
    # Sort filtered peaks by angle for the output
    filtered_peaks.sort(key=lambda x: x[0])
    
    return filtered_peaks

def plot_xray_data(df, sample_name, output_dir='figs'):
    """Create a plot of the X-ray data and save it to the output directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply a moving average smoothing with window size 5
    window_size = 5
    df_smoothed = df.copy()
    df_smoothed['Intensity (a.u.)'] = df['Intensity (a.u.)'].rolling(window=window_size, center=True).mean()
    
    # Find significant peaks
    peak_data = find_significant_peaks(df)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot raw data
    plt.plot(df['Angle (degrees)'], df['Intensity (a.u.)'], label='Raw Data', linewidth=1)
    
    # Plot smoothed data (skip NaN values at the beginning and end from rolling window)
    plt.plot(df_smoothed['Angle (degrees)'], df_smoothed['Intensity (a.u.)'], 
             label=f'Smoothed (window={window_size})', linewidth=2)
    
    # Annotate peaks if there are less than 10
    if len(peak_data) < 10:
        for angle, intensity in peak_data:
            plt.annotate(f'{angle:.1f}°', 
                         xy=(angle, intensity),
                         xytext=(0, 15),
                         textcoords='offset points',
                         ha='center',
                         fontsize=9,
                         arrowprops=dict(arrowstyle='->', lw=1, color='red'))
    
    # Add labels and title
    plt.xlabel('2θ (degrees)', fontsize=12)
    plt.ylabel('Intensity (a.u.)', fontsize=12)
    plt.title(f'XRD Pattern for {sample_name}', fontsize=14)
    plt.legend()
    
    # Set tight layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'{sample_name}_xrd.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    
    # Save peak data to a text file
    if peak_data:
        peak_file = os.path.join(output_dir, f'{sample_name}_peaks.txt')
        with open(peak_file, 'w') as f:
            f.write(f"Peak data for {sample_name}:\n")
            f.write("Angle (2θ)  |  Intensity\n")
            f.write("-" * 30 + "\n")
            for angle, intensity in sorted(peak_data, key=lambda x: x[0]):
                f.write(f"{angle:10.2f}  |  {intensity:10.2f}\n")
        print(f"Peak data saved to: {peak_file}")

def main():
    # Try different directories to find the CSV files
    possible_dirs = [
        "x_ray/data",                # Original path
        os.path.join(os.getcwd()),   # Current directory
        os.path.join(os.path.dirname(os.getcwd()), "data")  # Parent directory's data folder
    ]
    
    csv_files = []
    data_dir = None
    
    # Try each directory until we find CSV files
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            temp_csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
            if temp_csv_files:
                csv_files = temp_csv_files
                data_dir = dir_path
                print(f"Found {len(csv_files)} CSV files in {dir_path}")
                break
    
    if not csv_files:
        print("No CSV files found in any of the expected locations.")
        print("Current working directory:", os.getcwd())
        
        # Look for any CSV files in current directory recursively (up to 2 levels)
        for root, dirs, files in os.walk(os.getcwd()):
            depth = root[len(os.getcwd()):].count(os.sep)
            if depth <= 2:  # Limit search depth
                for file in files:
                    if file.endswith('.csv'):
                        csv_path = os.path.join(root, file)
                        csv_files.append(csv_path)
                        print(f"Found CSV file: {csv_path}")
        
        if not csv_files:
            print("Could not find any CSV files. Please check directory structure.")
            return
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "figs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for plots: {output_dir}")
    
    # Process each file
    for file_path in csv_files:
        try:
            df, sample_name = process_xray_file(file_path)
            plot_xray_data(df, sample_name, output_dir)
            print(f"Successfully processed {sample_name}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            # Print more detailed error information to help debug
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    main()