#!/bin/bash

if [[ "$1" ]]; then
    SHOP_UUID="$1"
else
    SHOP_UUID=$(python3 -c "
from utilities.io_utils import get_shop


shop_uuid = get_shop()[0]
print(shop_uuid)
")
fi


python3 -c "
from utilities.io_utils import build_database, fetch_person_data
import sys


shop_uuid = sys.argv[1]

build_database()
person_data = fetch_person_data(shop_uuid=shop_uuid)

print(person_data)
" "$SHOP_UUID"
