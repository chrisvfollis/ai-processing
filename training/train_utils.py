# standard dependencies
import os

# 3rd-party dependencies
from dotenv import load_dotenv
import psycopg2
import cv2
import torch
from torchvision import transforms
from torchvision.transforms import ConvertImageDtype
from torchvision.io import read_image

# internal dependencies
from utilities import io_utils


def get_approved_img_data(shop_id=None):
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
            results = cursor.fetchal()
            return results
    finally:
        conn.close()
