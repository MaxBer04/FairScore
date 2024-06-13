import os
import argparse
import shutil
import boto3
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))

def compress_dataset(dataset_dir, compressed_file):
    print(f"Compressing dataset directory '{dataset_dir}' to '{compressed_file}'...")
    shutil.make_archive(compressed_file, 'zip', os.path.dirname(dataset_dir), os.path.basename(dataset_dir))
    print("Dataset compressed successfully.")

def upload_to_s3(compressed_file, bucket_name, access_key_id, secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    print(f"Uploading '{compressed_file}' to S3 bucket '{bucket_name}'...")
    s3.upload_file(compressed_file + '.zip', bucket_name, os.path.basename(compressed_file) + '.zip')
    print("Dataset uploaded to S3 successfully.")

def download_from_s3(bucket_name, compressed_file, access_key_id, secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    print(f"Downloading '{compressed_file}' from S3 bucket '{bucket_name}'...")
    s3.download_file(bucket_name, compressed_file + '.zip', compressed_file + '.zip')
    print("Dataset downloaded from S3 successfully.")

def extract_dataset(compressed_file, dataset_dir):
    print(f"Extracting '{compressed_file}' to '{dataset_dir}'...")
    shutil.unpack_archive(compressed_file + '.zip', dataset_dir)
    print("Dataset extracted successfully.")

def main():
    parser = argparse.ArgumentParser(description='Dataset Compression and S3 Uploader/Downloader')
    parser.add_argument('action', choices=['upload', 'download'], help='Action to perform: upload or download')
    parser.add_argument('--dataset_dir', default='dataset', help='Directory containing the dataset')
    parser.add_argument('--compressed_file', default='d_RV2_5000_balanced', help='Name of the compressed file')
    parser.add_argument('--bucket_name', default="masterarbeit-2", help='Name of the S3 bucket')
    args = parser.parse_args()

    access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    dataset_path = os.path.join(script_dir, args.dataset_dir)

    if args.action == 'upload':
        compress_dataset(dataset_path, args.compressed_file)
        upload_to_s3(args.compressed_file, args.bucket_name, access_key_id, secret_access_key)
    elif args.action == 'download':
        download_from_s3(args.bucket_name, args.compressed_file, access_key_id, secret_access_key)
        extract_dataset(args.compressed_file, dataset_path)

if __name__ == '__main__':
    main()
