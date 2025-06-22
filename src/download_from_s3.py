import argparse
import os
from dotenv import dotenv_values
import boto3

BUCKET_NAME = 'pabd25'
YOUR_SURNAME = 'vvkorotkova'
FILE_PATH = 'models/best_model.joblib'
os.makedirs('models', exist_ok=True)
config = dotenv_values(".env")


def main(args):
    client = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key']
    )

    object_name = f'{YOUR_SURNAME}/' + args.input
    client.download_file(Bucket=BUCKET_NAME, 
                            Key=object_name, 
                            Filename=args.input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Input data files to download from S3 storage',
                        default=FILE_PATH)
    
    args = parser.parse_args()
    print(args)
    main(args)