import csv
import os
import sys

def convert_txt_to_csv(filename):
    """
    Takes a .txt file, removes the top row, and splits the remaining content 
    into two comma-separated columns in a CSV file with the same base name.
    
    Args:
        filename (str): Path to the input .txt file
    """
    # Ensure the input file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return
    
    # Ensure it's a .txt file
    if not filename.lower().endswith('.txt'):
        print(f"Error: File '{filename}' is not a .txt file.")
        return
    
    # Define output filename with .csv extension
    output_filename = os.path.splitext(filename)[0] + '.csv'
    
    try:
        # Read the input file
        with open(filename, 'r') as txt_file:
            # Skip the first line (header)
            next(txt_file, None)
            
            # Read the rest of the lines
            lines = [line.strip() for line in txt_file if line.strip()]
        
        # Write to CSV
        with open(output_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Process each line and split into two columns
            for line in lines:
                # Split the line in half
                mid_point = len(line) // 2
                first_column = line[:mid_point].strip()
                second_column = line[mid_point:].strip()
                
                # Write to CSV
                writer.writerow([first_column, second_column])
        
        print(f"Successfully converted '{filename}' to '{output_filename}'")
        print(f"Removed top row and split remaining content into two columns.")
    
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    # Check if filename was provided as command line argument
    if len(sys.argv) < 2:
        print("Usage: python txt_to_csv.py <filename.txt>")
    else:
        convert_txt_to_csv(sys.argv[1])