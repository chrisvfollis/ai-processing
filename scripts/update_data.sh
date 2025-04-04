#!/bin/bash

if [[ "$1" ]]; then
    SHOP_UUID="$1"
else
    SHOP_UUID=$(python3 -c "
from utilities.io_utils import get_shop
print(get_shop(db_path='files/data.db')[0])
")
fi


RESULT=$(python3 -c "
from utilities.io_utils import build_database, fetch_person_data
import sys

db_path = 'files/data.db'
img_dir = 'files/input/faces'

build_database(db_path=db_path)

shop_uuid = sys.argv[1]
print(fetch_person_data(shop_uuid=shop_uuid, db_path=db_path, img_dir=img_dir))
" "$SHOP_UUID")
