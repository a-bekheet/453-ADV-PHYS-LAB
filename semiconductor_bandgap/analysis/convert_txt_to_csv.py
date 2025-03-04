import csv
import os
import sys
import re

def convert_txt_to_csv(filename):
    """
    Takes a .txt file, removes the top row, and converts the remaining content
    into a two-column CSV with temperature and resistance values.

    This is especially useful for the Energy Gap VI data files.
    
    Args:
        filename (str): Path to the input .txt file
        
    Returns:
        str: Path to the created CSV file, or None if an error occurred
    """
    # Ensure the input file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None
    
    # Ensure it's a .txt file
    if not filename.lower().endswith('.txt'):
        print(f"Error: File '{filename}' is not a .txt file.")
        return None
    
    # Define output filename with .csv extension
    output_filename = os.path.splitext(filename)[0] + '.csv'
    
    try:
        # Read the input file
        with open(filename, 'r') as txt_file:
            lines = txt_file.readlines()
        
        # Skip the first line (header)
        data_lines = lines[1:] if lines else []
        
        # Extract data
        processed_data = []
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to find numeric values in the line
            # Look for patterns that might be temperature and resistance values
            numbers = re.findall(r'\b\d+\.?\d*\b', line)
            
            if len(numbers) >= 2:
                # Assume first number is temperature, second is resistance
                temp = numbers[0]
                res = numbers[1]
                processed_data.append([temp, res])
            else:
                # Alternative: split by whitespace and try to extract two values
                parts = line.split()
                if len(parts) >= 2:
                    # Try to convert parts to float to verify they're numbers
                    try:
                        temp = float(parts[0])
                        res = float(parts[1])
                        processed_data.append([str(temp), str(res)])
                    except ValueError:
                        # If conversion fails, try another approach
                        # For example, split the line in half
                        mid_point = len(line) // 2
                        first_column = line[:mid_point].strip()
                        second_column = line[mid_point:].strip()
                        
                        # Try to find numbers in each half
                        first_number = re.search(r'\b\d+\.?\d*\b', first_column)
                        second_number = re.search(r'\b\d+\.?\d*\b', second_column)
                        
                        if first_number and second_number:
                            processed_data.append([first_number.group(), second_number.group()])
        
        # Write to CSV
        with open(output_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Add header row for clarity
            writer.writerow(["Temperature", "Resistance"])
            
            # Write the processed data
            for row in processed_data:
                writer.writerow(row)
        
        print(f"Successfully converted '{filename}' to '{output_filename}'")
        print(f"Extracted {len(processed_data)} data points with temperature and resistance values.")
        
        # Return the path to the created CSV file
        return output_filename
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    # Check if filename was provided as command line argument
    if len(sys.argv) < 2:
        print("Usage: python convert_txt_to_csv.py <filename.txt>")
    else:
        csv_path = convert_txt_to_csv(sys.argv[1])
        if csv_path:
            print(f"CSV file created at: {csv_path}")