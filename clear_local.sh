#!/bin/bash

rm -f files/output/*.pkl
rm -f files/output/*.hdf5

rm -f files/output/runtime_data/*.xlsx
rm -f files/output/videos/*.mp4

journalctl --rotate
journalctl --vacuum-time=1s
