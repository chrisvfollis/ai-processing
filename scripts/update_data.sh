#!/bin/bash

if [[ "$1" ]]; then
    SHOP_UUID="$1"
else
    SHOP_UUID=$(python3 -c "
    from utilities.io_utils import get_shop
    print(get_shop()[0])
    ")
fi


RESULT=$(python3 -c "
from utilities.io_utils import build_database, fetch_person_data
import sys

build_database()

shop_uuid = sys.argv[1]
print(fetch_person_data(shop_uuid=shop_uuid))
" "$SHOP_UUID")
