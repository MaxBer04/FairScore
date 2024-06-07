import os
import pandas as pd
import pyarrow.parquet as pq
import argparse

def extract_images(parquet_dir):
    # Ensure the directory exists
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"The directory {parquet_dir} does not exist.")
    
    # Create 'images' directory if it doesn't exist
    images_dir = os.path.join(parquet_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Iterate over all parquet files in the directory
    for file_name in os.listdir(parquet_dir):
        if file_name.endswith('.parquet'):
            file_path = os.path.join(parquet_dir, file_name)
            print(f"Processing {file_path}...")

            # Read the parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()

            # Print the columns to check the structure
            print(f"Columns in {file_name}: {df.columns.tolist()}")

            # Assuming images are stored in a column named 'image'
            if 'image' not in df.columns:
                print(f"'image' column not found in {file_name}. Skipping...")
                continue

            for index, row in df.iterrows():
                image_data = row['image']['bytes']
                # Assuming images are stored as bytes
                # Convert and save the image
                image_file_path = os.path.join(images_dir, f"image_{index}.jpg")
                with open(image_file_path, 'wb') as img_file:
                    img_file.write(image_data)
    
    print(f"Images extracted to {images_dir}")

def inspect_parquet(parquet_file):
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"The file {parquet_file} does not exist.")
    
    table = pq.read_table(parquet_file)
    print(f"Schema of {parquet_file}:")
    print(table.schema)

def main():
    parser = argparse.ArgumentParser(description="Utility script for Parquet files.")
    parser.add_argument('action', type=str, choices=['extract', 'inspect'], help="Action to perform: 'extract' or 'inspect'.")
    parser.add_argument('parquet_path', type=str, help="Path to the parquet file or directory.")
    args = parser.parse_args()

    if args.action == 'extract':
        extract_images(args.parquet_path)
    elif args.action == 'inspect':
        inspect_parquet(args.parquet_path)

if __name__ == "__main__":
    main()
