#!/bin/sh

#rm -rf train.log

TIME=`date`

echo "###########ROUND_START:$TIME" >>train.log 2>&1

echo GBR >>train.log 2>&1
python gbr.py  >>train.log 2>&1 

#echo RFR >>train.log 2>&1
#python rfr.py  >>train.log 2>&1 

#echo SVR >>train.log 2>&1
#python svr.py >>train.log 2>&1  

#echo XGB >>train.log 2>&1
#python xgb.py >>train.log 2>&1

#echo lightGBM >>train.log 2>&1
#python lgb.py >>train.log 2>&1

echo "###########ROUND_END:$TIME" >>train.log 2>&1

egrep 'ROUND|Score|MSE' train.log
