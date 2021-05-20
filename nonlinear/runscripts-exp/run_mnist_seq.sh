#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1
MNIST_SIZE="20x20"

BIN=../source/mnist_hqrc.py
SAVE=/data/zoro/hqrc/mnist
STRENGTHS='0.0,0.1,0.3,0.5,0.7,0.9'
FULL=1
LB1=3
LB2=7
CORR=0
NQR=20

for V in 1 5 10 20
do
python $BIN --mnist_size $MNIST_SIZE --savedir $SAVE --nqrs $NQR --virtuals $V --full $FULL --label1 $LB1 --label2 $LB2 --strengths $STRENGTHS --use_corr $CORR
done
