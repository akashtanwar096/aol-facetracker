import boto3
import pandas as pd

AWS_KEY_CODE_PATH = "aws_key_code.txt"

def get_s3_client():
    keycodedf = pd.read_csv(AWS_KEY_CODE_PATH, header=None)
    AWS_ID = keycodedf.iloc[0][0]
    AWS_SECRET_KEY = keycodedf.iloc[1][0]
    session = boto3.Session(aws_access_key_id=AWS_ID,aws_secret_access_key=AWS_SECRET_KEY, region_name='ap-south-1')
    s3 = session.client('s3')
    return s3