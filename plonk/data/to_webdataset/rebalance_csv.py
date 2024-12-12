import csv
import os
import sys
import glob
import tqdm


def split_csv_files(input_files, output_dir, lines_per_file=100000):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize counters
    total_lines = 0
    file_count = 0
    current_line_count = 0

    # Initialize the first output file
    output_file = os.path.join(output_dir, f"{str(file_count).zfill(3)}.csv")
    output_writer = open(output_file, "w", newline="")
    csv_writer = None

    try:
        for file_path in tqdm.tqdm(input_files, desc="Processing files"):
            with open(file_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file)

                # Initialize writer once we have the header row
                if csv_writer is None:
                    header = next(csv_reader)
                    csv_writer = csv.writer(output_writer)
                    csv_writer.writerow(header)

                # Process each line in the current file
                for row in csv_reader:
                    if current_line_count >= lines_per_file:
                        # Close the current file and start a new one
                        output_writer.close()
                        file_count += 1
                        current_line_count = 0
                        output_file = os.path.join(
                            output_dir, f"{str(file_count).zfill(3)}.csv"
                        )
                        output_writer = open(output_file, "w", newline="")
                        csv_writer = csv.writer(output_writer)
                        csv_writer.writerow(header)  # Write header to new file

                    # Write row to the current output file
                    csv_writer.writerow(row)
                    current_line_count += 1
                    total_lines += 1

    finally:
        # Close the last output file
        if output_writer:
            output_writer.close()

    print(f"Total lines processed: {total_lines}")
    print(f"Files created: {file_count + 1}")


if __name__ == "__main__":
    input_dir = "../datasets/YFCC100M/yfcc100m_dataset_with_gps_train"
    output_dir = "../datasets/YFCC100M/yfcc100m_dataset_with_gps_train_balanced"
    lines_per_file = 100000

    # Get all CSV files in input directory
    input_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not input_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} CSV files")
    split_csv_files(input_files, output_dir, lines_per_file)
