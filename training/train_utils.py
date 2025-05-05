# standard dependencies
import os
from io import BytesIO

# 3rd-party dependencies
from dotenv import load_dotenv
import psycopg2
import boto3
from PIL import Image
import pandas as pd
import cv2
import torch
from torchvision import transforms
from torchvision.transforms import ConvertImageDtype
from torchvision.io import read_image

# internal dependencies
from utilities import io_utils


def get_approved_records(shop_id=None):
    ''''
    Retrieve approved employee event records to get accurate labeled image data
    for training and/or fine tuning.
    '''
    
    conn = io_utils.pg_db_connect(var_prefix='WEBAPP_DB')

    query = """
        SELECT eel.employee_id, eel.shop_id, eel.start_time, eel.image,
               sel.first_name, sel.last_name
        FROM employee_event_log_employeeevent eel
        JOIN shop_employeelist sel ON eel.employee_id = sel.id
        WHERE eel.review_status = 'approved'
    """
    params = ()
    if shop_id is not None:
        query += ' AND eel.shop_id = %s'
        params = (shop_id,)
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            return results
    finally:
        conn.close()


def get_approved_img_data(approved_records, bucket_name='timemanager-event-imgs',
                          output_dir = '../files/output/'):
    s3 = boto3.client('s3')
    img_output_dir = os.path.join(output_dir, 'event_imgs/')

    saved_image_data = []

    for row in approved_records:
        employee_id, shop_id, start_time, image_key, first_name, last_name = row

        try:
            s3_obj = s3.get_object(Bucket=bucket_name, Key=image_key)
            img = Image.open(BytesIO(s3_obj['Body'].read()))

            save_path = os.path.join(img_output_dir, image_key)

            img.save(save_path)

            saved_image_data.append({
                'employee_id': employee_id,
                'shop_id': shop_id,
                'image_key': image_key,
                'first_name': first_name,
                'last_name': last_name,
                'start_time': start_time,
            })

            img_data_df = pd.DataFrame(saved_image_data)
            excel_path = os.path.join(output_dir, 'approved_img_data.xlsx')

            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                img_data_df.to_excel(writer, sheet_name='Approved Image Data', index=False)


        except Exception as e:
            print(f'Error retrieving or saving image {image_key}: {e}')

    return saved_image_data
