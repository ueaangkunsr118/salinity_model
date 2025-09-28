import csv

def remove_columns(input_file, output_file, columns_to_remove):
    with open(input_file, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = [field for field in reader.fieldnames if field not in columns_to_remove]
        
        with open(output_file, mode='w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                for column in columns_to_remove:
                    del row[column]
                writer.writerow(row)

# Example usage
remove_columns('ladpo_hourlyw.csv', 'ladpo_hourlyw_mod.csv', ['Temperature', 'Conductivity','TDS','Sensor_Depth','relative_humidity','dew_point','sealevel_pressure','surface_pressure','wind_speed_10m','wind_speed_100m'])