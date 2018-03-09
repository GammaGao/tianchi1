#!/bin/sh
python date_map.py
python create_features.py
rm -rf train_tmp.csv  testa_tmp.csv testb_tmp.csv datesmap.csv id_date.csv
