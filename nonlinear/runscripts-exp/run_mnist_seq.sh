#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1
MNIST_SIZE="14x14"

BIN=../source/mnist_hqrc.py
SAVE=/data/zoro/hqrc/mnist
STRENGTHS='0.5'

FULL=1
LB1=3
LB2=5
CORR=0
NQR=14
TAUS='3.0'
LN=0
TRANS=100

for V in 5
do
python $BIN --transient $TRANS --linear_reg $LN --taudeltas $TAUS --mnist_size $MNIST_SIZE --savedir $SAVE --nqrs $NQR --virtuals $V --full $FULL --label1 $LB1 --label2 $LB2 --strengths $STRENGTHS --use_corr $CORR
done
