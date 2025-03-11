import os
import re
import csv
import pandas as pd
import sys
import argparse

# Import the convert_txt_to_csv function from the existing script
from convert_txt_to_csv import convert_txt_to_csv

def extract_run_info(filename):
    """
    Extract experiment type and run number from filename.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        tuple: (experiment_type, run_number) or None if pattern doesn't match
    """
    # Pattern matches: date_type_runN.txt or date_type_testN.txt
    pattern = r'([a-zA-Z0-9]+)_([a-zA-Z]+)_(?:run|test)(\d+)\.txt$'
    
    match = re.match(pattern, os.path.basename(filename))
    if match:
        date, exp_type, run_num = match.groups()
        return f"{date}_{exp_type}", run_num
    return None, None

def process_file(filepath):
    """
    Convert a TXT file to CSV format and return the path to the CSV.
    
    Args:
        filepath (str): Path to the TXT file
        
    Returns:
        str: Path to the CSV file or None if conversion failed
    """
    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None
        
    # Ensure it's a .txt file
    if not filepath.lower().endswith('.txt'):
        print(f"Error: File '{filepath}' is not a .txt file.")
        return None
    
    # Check if corresponding CSV already exists
    csv_path = os.path.splitext(filepath)[0] + '.csv'
    
    # If CSV doesn't exist, create it using the convert function
    if not os.path.exists(csv_path):
        print(f"Converting {os.path.basename(filepath)} to CSV...")
        csv_path = convert_txt_to_csv(filepath)
        if not csv_path:
            print(f"Error converting {os.path.basename(filepath)}. Skipping.")
            return None
    else:
        print(f"Using existing CSV file: {os.path.basename(csv_path)}")
    
    return csv_path

def aggregate_specified_files(output_file, file_list):
    """
    Aggregate data from a specified list of files.
    
    Args:
        output_file (str): Path for the output aggregated CSV file
        file_list (list): List of file paths to aggregate
        
    Returns:
        str: Path to the aggregated CSV file or None if failed
    """
    all_data = []
    
    # Process each file
    for filepath in file_list:
        # Convert TXT to CSV if needed
        csv_path = process_file(filepath)
        if not csv_path:
            continue
        
        try:
            # Extract run info from filename
            _, run_number = extract_run_info(filepath)
            run_label = f"Run {run_number}" if run_number else f"Run (unknown)"
            
            # Read the CSV
            df = pd.read_csv(csv_path)
            
            # Add run identifier column
            df['Run'] = run_label
            
            # Add to our collection
            all_data.append(df)
            print(f"Added data from {os.path.basename(csv_path)}")
            
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
    
    if not all_data:
        print("No valid data found in the specified files")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Created aggregated file: {output_file}")
    
    return output_file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Aggregate temperature and resistance data from specified TXT files')
    parser.add_argument('output', help='Name of the output aggregated CSV file')
    parser.add_argument('files', nargs='+', help='List of TXT files to process')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    files = [os.path.abspath(f) for f in args.files]
    output_file = os.path.abspath(args.output)
    
    print(f"Will process {len(files)} files and save as: {output_file}")
    
    # Process files
    result = aggregate_specified_files(output_file, files)
    
    if result:
        print("\nAggregation complete!")
    else:
        print("\nAggregation failed.")

if __name__ == "__main__":
    main()