import os
import argparse
import shutil
import boto3
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

def compress_dataset(dataset_dir, compressed_file):
    print(f"Compressing dataset directory '{dataset_dir}' to '{compressed_file}'...")
    
    def make_archive(base_name, format, root_dir, base_dir):
        shutil.make_archive(base_name, format, root_dir, base_dir)
    
    total_files = sum(len(files) for _, _, files in os.walk(dataset_dir))
    with tqdm(total=total_files, unit='file', desc='Compressing') as pbar:
        make_archive(compressed_file, 'zip', os.path.dirname(dataset_dir), os.path.basename(dataset_dir))
        pbar.update(total_files)
    
    print("Dataset compressed successfully.")

def upload_to_s3(compressed_file, bucket_name, access_key_id, secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    file_size = os.path.getsize(compressed_file + '.zip')
    print(f"Uploading '{compressed_file}' to S3 bucket '{bucket_name}'...")
    with tqdm(total=file_size, unit='B', unit_scale=True, desc='Uploading') as pbar:
        s3.upload_file(compressed_file + '.zip', bucket_name, os.path.basename(compressed_file) + '.zip', Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    print("Dataset uploaded to S3 successfully.")

def upload_directory_to_s3(directory, bucket_name, access_key_id, secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    total_files = sum(len(files) for _, _, files in os.walk(directory))
    root_folder_name = os.path.basename(directory.rstrip('/'))
    with tqdm(total=total_files, unit='file', desc='Uploading Directory') as pbar:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                s3_path = os.path.join(root_folder_name, os.path.relpath(file_path, directory))
                s3.upload_file(file_path, bucket_name, s3_path, Callback=lambda bytes_transferred: pbar.update(1))
    print("Directory uploaded to S3 successfully.")

def download_from_s3(bucket_name, compressed_file, access_key_id, secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    print(f"Downloading '{compressed_file}' from S3 bucket '{bucket_name}'...")
    with tqdm(unit='B', unit_scale=True, desc='Downloading') as pbar:
        s3.download_file(bucket_name, compressed_file + '.zip', compressed_file + '.zip', Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
    print("Dataset downloaded from S3 successfully.")

def extract_dataset(compressed_file, dataset_dir):
    print(f"Extracting '{compressed_file}' to '{dataset_dir}'...")
    
    def unpack_archive(filename, extract_dir):
        shutil.unpack_archive(filename, extract_dir)
    
    with tqdm(unit='file', desc='Extracting') as pbar:
        unpack_archive(compressed_file + '.zip', dataset_dir)
        total_files = sum(len(files) for _, _, files in os.walk(dataset_dir))
        pbar.update(total_files)
    
    print("Dataset extracted successfully.")

def download_directory_from_s3(bucket_name, directory, access_key_id, secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    print(f"Downloading directory '{directory}' from S3 bucket '{bucket_name}'...")
    root_folder_name = os.path.basename(directory.rstrip('/'))
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=root_folder_name + '/')
    
    total_files = sum(1 for page in pages for _ in page.get('Contents', []))
    pages = paginator.paginate(Bucket=bucket_name, Prefix=root_folder_name + '/')
    
    with tqdm(total=total_files, unit='file', desc='Downloading Directory') as pbar:
        for page in pages:
            for obj in page.get('Contents', []):
                s3_path = obj['Key']
                local_path = os.path.join(directory, os.path.relpath(s3_path, root_folder_name))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket_name, s3_path, local_path)
                pbar.update(1)
    print("Directory downloaded from S3 successfully.")

def main():
    parser = argparse.ArgumentParser(description='Dataset Compression and S3 Uploader/Downloader')
    parser.add_argument('action', choices=['upload', 'download'], help='Action to perform: upload or download')
    parser.add_argument('--dataset_dir', default='output-50k-FF-09T-5it', help='Directory containing the dataset')
    parser.add_argument('--compressed_file', default='ms_RV2_37k_gender_FF-Face---09T-5it', help='Name of the compressed file')
    parser.add_argument('--bucket_name', default="masterarbeit-2", help='Name of the S3 bucket')
    parser.add_argument('--no_compress', action='store_true', help='Upload or download the directory without compression')
    args = parser.parse_args()

    access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    dataset_path = os.path.join(script_dir, args.dataset_dir)

    if args.action == 'upload':
        if args.no_compress:
            upload_directory_to_s3(dataset_path, args.bucket_name, access_key_id, secret_access_key)
        else:
            compress_dataset(dataset_path, args.compressed_file)
            upload_to_s3(args.compressed_file, args.bucket_name, access_key_id, secret_access_key)
    elif args.action == 'download':
        if args.no_compress:
            download_directory_from_s3(args.bucket_name, dataset_path, access_key_id, secret_access_key)
        else:
            download_from_s3(args.bucket_name, args.compressed_file, access_key_id, secret_access_key)
            extract_dataset(args.compressed_file, dataset_path)

if __name__ == '__main__':
    main()
