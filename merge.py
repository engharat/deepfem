import csv
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

csv_files = [
    f for f in os.listdir(script_dir)
    if f.lower().endswith(".csv") and f != "Simulations_Info.csv"
]

header = None
unique_rows = set()   # to track seen rows
merged_rows = []      # to preserve row data

for filename in csv_files:
    file_path = os.path.join(script_dir, filename)
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        file_header = next(reader)

        # Save header only once
        if header is None:
            header = file_header

        for row in reader:
            if row:
                row_tuple = tuple(row)
                if row_tuple not in unique_rows:
                    unique_rows.add(row_tuple)
                    merged_rows.append(row)

# Sort rows numerically using the first column (integer index)
merged_rows.sort(key=lambda x: int(x[0]))

# Write the merged CSV
output_path = os.path.join(script_dir, "Simulations_Info.csv")
with open(output_path, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)
    writer.writerows(merged_rows)

print(f"Merged {len(csv_files)} CSV files into {output_path}")
print(f"Total unique rows: {len(merged_rows)}")
