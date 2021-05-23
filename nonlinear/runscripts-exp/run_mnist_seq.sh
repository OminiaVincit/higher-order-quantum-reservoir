#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1
MNIST_SIZE="20x20"

BIN=../source/mnist_hqrc.py
SAVE=/data/zoro/hqrc/mnist

FULL=1
LB1=3
LB2=5
CORR=0
NQR=10
LN=0
TRANS=100
RATE=0.1
N=1
STRENS='0.0,0.1,0.3,0.5,0.7,0.9'
#TAUS='3.0'
#--strengths $STRENGTHS --taudeltas $TAUS

for V in 5 10
do
python $BIN --strengths $STRENS --ntrials $N --transient $TRANS --rate $RATE --linear_reg $LN --mnist_size $MNIST_SIZE --savedir $SAVE --nqrs $NQR --virtuals $V --full $FULL --label1 $LB1 --label2 $LB2 --use_corr $CORR
done
