# standard dependencies
import os

# 3rd-party dependencies
import boto3
from dotenv import load_dotenv
import psycopg2
import sqlite3
import requests

# internal dependencies
pass


# =============================================================================
#                                  - APIS -
# -----------------------------------------------------------------------------


class APIClient:
    def __init__(self, var_prefix: str = 'INTERNAL_API'):
        load_dotenv()
        self.base_url = os.getenv(f'{var_prefix}_URL')
        self.api_key = os.getenv(f'{var_prefix}_KEY')

        if (not self.base_url) or (not self.api_key):
            missing = ', '.join([
                key for key, value in [
                    (f'{var_prefix}_URL', self.base_url),
                    (f'{var_prefix}_KEY', self.api_key)
                ] if not value
            ])
            raise ValueError(f'{missing} not found in .env')
        
        self.headers = {
            'X-Custom-API-Key': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def endpoint_url(self, endpoint: str):
        return self.base_url + endpoint
    
    def get(
            self, endpoint: str, headers: dict = None, params: dict = None,
            json=None
        ) -> requests.models.Response:

        endpoint_url = self.endpoint_url(endpoint)
        headers = headers or self.headers

        return requests.get(endpoint_url, headers=headers, params=params, json=json)
    
    def post(
            self, endpoint: str, headers: dict = None, params: dict = None,
            json=None
        ) -> requests.models.Response:

        endpoint_url = self.endpoint_url(endpoint)
        headers = headers or self.headers

        return requests.post(endpoint_url, headers=headers, params=params, json=json)


# =============================================================================
#                              - SQL DATABASES -
# -----------------------------------------------------------------------------


def sqlite_db_connect(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    return conn, cursor


def pg_db_connect(conn_args=None, var_prefix='PG'):
    conn_params = ['host', 'port', 'user', 'password', 'dbname']
    if not conn_args:
        load_dotenv()

        conn_args = [
            os.getenv(f'{var_prefix}_HOST'),
            os.getenv(f'{var_prefix}_PORT'),
            os.getenv(f'{var_prefix}_USER'),
            os.getenv(f'{var_prefix}_PASSWORD'),
            os.getenv(f'{var_prefix}_NAME')
        ]
    conn_details = dict(zip(conn_params, conn_args))

    conn = psycopg2.connect(**conn_details)
    cursor = conn.cursor()

    return conn, cursor


def close_pg_db(conn, cursor):
    cursor.close()  # best practice to explicitly close the cursor
    conn.close()


def close_sqlite_db(conn, cursor, commit: bool = False):
    if commit:
        conn.commit()

    cursor.close()  # best practice to explicitly close the cursor
    conn.close()


# =============================================================================
#                              - AWS RESOURCES -
# -----------------------------------------------------------------------------


def get_aws_credentials():
    load_dotenv()
    access_key = os.environ.get('AWS_ACCESS_KEY')
    secret_key = os.environ.get('AWS_SECRET_KEY')

    return access_key, secret_key


def s3_connect(region='us-west-1', credentials=None):
    access_key, secret_key = credentials or get_aws_credentials()

    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    return s3_client


def ec2_connect(region='us-west-1', credentials=None):
    access_key, secret_key = credentials or get_aws_credentials()

    s3_client = boto3.client(
        'ec2',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    return s3_client
