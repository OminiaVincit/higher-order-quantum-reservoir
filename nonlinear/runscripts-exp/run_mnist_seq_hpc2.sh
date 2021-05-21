#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1
MNIST_SIZE="28x28"

BIN=../source/mnist_hqrc.py
SAVE=../../../data/hqrc/mnist
STRENGTHS='0.0,0.1,0.3,0.5,0.7,0.9'
FULL=0
LB1=0
LB2=1
CORR=0
LN=0

#STRENGTHS='0.0'
#TAUS='0.0'
#--taudeltas $TAUS 

for NQR in 28
do
for V in 5 10 20
do
python $BIN --linear_reg $LN --mnist_size $MNIST_SIZE --savedir $SAVE --nqrs $NQR --virtuals $V --full $FULL --label1 $LB1 --label2 $LB2 --strengths $STRENGTHS --use_corr $CORR
done
done
