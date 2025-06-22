import argparse
import os
from dotenv import dotenv_values
import boto3

BUCKET_NAME = 'pabd25'
YOUR_SURNAME = 'vvkorotkova'
LOCAL_FILE_PATH = ['models/best_model_2025-05-26_21-09.joblib']

def main(args):
    try:
        print("Loading AWS credentials...")
        config = dotenv_values(".env")
        
        print("Initializing S3 client...")
        client = boto3.client(
            's3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id=config['aws_access_key_id'],
            aws_secret_access_key=config['aws_secret_access_key']
        )

        for file_path in args.input:
            print(f"\nProcessing file: {file_path}")
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} not found!")
                continue

            object_name = f'{YOUR_SURNAME}/' + 'models/best_model.joblib'
            print(f"Uploading to S3 path: {BUCKET_NAME}/{object_name}")
            
            client.upload_file(file_path, BUCKET_NAME, object_name)
            print("Upload successful!")

        print("\nAll operations completed.")
    except Exception as e:
        print(f"\nCritical error: {str(e)}")

if __name__ == '__main__':
    print("Starting S3 upload script...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+',
                      help='Input local data files to upload to S3 storage',
                      default=LOCAL_FILE_PATH)
    args = parser.parse_args()
    
    print(f"Files to upload: {args.input}")
    main(args)