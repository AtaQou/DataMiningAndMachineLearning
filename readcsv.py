import csv

# Input and output file paths
input_file = 'data.csv'
output_file = '10lines.csv'

# Number of rows to extract (including header)
num_rows = 10

with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for i, row in enumerate(reader):
        if i >= num_rows:
            break
        writer.writerow(row)

print(f"✅ Done! First {num_rows} lines written to '{output_file}'")
