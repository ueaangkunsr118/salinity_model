import os
import pandas as pd
import matplotlib.pyplot as plt

# Input and output directories
input_dir = "/Users/ratchanonkhongsawi/Desktop/CMKL/SELF-Learn/RND1/Data/clean"  # Folder with cleaned CSV files
plot_dir = "/Users/ratchanonkhongsawi/Desktop/CMKL/SELF-Learn/RND1/Data/plot"  # Folder to save the plots

# Ensure plot directory exists
os.makedirs(plot_dir, exist_ok=True)

# Loop through all CSV files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(input_dir, filename)
        plot_file = os.path.join(plot_dir, filename.replace(".csv", ".png"))

        try:
            # Read the cleaned CSV file
            df = pd.read_csv(file_path)

            # Convert 'date_time' to datetime format if it exists
            if 'date_time' in df.columns and 'Salinity(g/L)' in df.columns:
                df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
                df.set_index('date_time', inplace=True)

                # Plot Salinity(g/L) over time
                fig, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(df.index, df['Salinity(g/L)'], '.', markersize=1, label='Salinity(g/L)', color='b')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Salinity(g/L) (g/L)', color='b')
                ax1.tick_params(axis='y', labelcolor='b')

                # Title and save the plot
                plt.title(f'Salinity(g/L) Over Time: {filename}')
                plt.tight_layout()
                plt.savefig(plot_file)
                plt.close()
                
                print(f"Plot saved: {plot_file}")
            else:
                print(f"'date_time' or 'Salinity(g/L)' column missing in {filename}, skipping.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("All plots have been generated and saved.")