# standard dependencies
import os

# 3rd-party dependencies
import boto3
from dotenv import load_dotenv
import psycopg2
import sqlite3

# internal dependencies
from utilities import io_utils


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


def s3_connect(region='us-west-1', credentials=None):
    access_key, secret_key = credentials or io_utils.get_aws_credentials()

    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    return s3_client


def ec2_connect(region='us-west-1', credentials=None):
    access_key, secret_key = credentials or io_utils.get_aws_credentials()

    s3_client = boto3.client(
        'ec2',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    return s3_client
