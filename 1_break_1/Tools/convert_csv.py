import os
import pandas as pd

# Define input and output directory
input_dir = "/Users/ratchanonkhongsawi/Desktop/CMKL/SELF-Learn/RND1/Data/unclean_exel"  # Path to the folder containing .xlsx files
output_dir = "/Users/ratchanonkhongsawi/Desktop/CMKL/SELF-Learn/RND1/Data/unclean_csv"  # Path where .csv files will be saved

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".xlsx"):  # Check if the file has .xlsx extension
        # Construct full file paths
        xlsx_path = os.path.join(input_dir, filename)
        csv_filename = filename.replace(".xlsx", ".csv")
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Read the .xlsx file
        df = pd.read_excel(xlsx_path)
        
        # Save as .csv
        df.to_csv(csv_path, index=True)
        
        print(f"Converted: {xlsx_path} --> {csv_path}")

print("All .xlsx files have been converted to .csv.")