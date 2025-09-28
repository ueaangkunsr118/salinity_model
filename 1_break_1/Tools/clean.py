import os
import pandas as pd

# Define input and output directories
input_dir = "/Users/ratchanonkhongsawi/Desktop/CMKL/SELF-Learn/RND1/Data/unclean_csv"  # Path to folder containing CSV files
output_dir = "/Users/ratchanonkhongsawi/Desktop/CMKL/SELF-Learn/RND1/Data/clean"  # Path to store cleaned CSV files

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(input_dir, filename)
        try:
            # Read CSV, fixing broken headers and missing values
            df = pd.read_csv(file_path, skip_blank_lines=True)

            # Clean up column names: remove special characters, line breaks, and spaces
            df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('"', '').str.replace(' ', '_')
            
            # Check if 'Salinity' column exists (cleaned name: Salinity(g/L))
            if 'Salinity(g/L)' in df.columns:
                # Filter rows where Salinity >= 0 and drop rows with missing Salinity
                filtered_df = df[(pd.to_numeric(df['Salinity(g/L)'], errors='coerce') >= 0) & 
                 (pd.to_numeric(df['Salinity(g/L)'], errors='coerce') <= 100)]
                
                # Save cleaned and filtered data to the output folder
                output_file_path = os.path.join(output_dir, filename)
                filtered_df.to_csv(output_file_path, index=False)
                print(f"Cleaned file saved: {output_file_path}")
            else:
                print(f"'Salinity(g/L)' column not found in {filename}, skipping.")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

print("Data cleaning complete.")