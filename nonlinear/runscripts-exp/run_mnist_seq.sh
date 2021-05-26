#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1
MNIST_SIZE="14x14"

BIN=../source/mnist_hqrc.py
SAVE=/data/zoro/hqrc/mnist

FULL=1
LB1=3
LB2=5
CORR=0
NQR=14
LN=0
TRANS=100
N=8
DYNAMIC='phase_trans'
#'full_random,half_random,full_const_trans,full_const_coeff,ion_trap,phase_trans'
STRENS='0.5'
DS='0.0'
#TAUS='3.0'
#--strengths $STRENGTHS --taudeltas $TAUS

for RATE in 0.1 0.5 1.0
do
for V in 20
do
python $BIN --dynamic $DYNAMIC --non_diags $DS --strengths $STRENS --ntrials $N --transient $TRANS --rate $RATE --linear_reg $LN --mnist_size $MNIST_SIZE --savedir $SAVE --nqrs $NQR --virtuals $V --full $FULL --label1 $LB1 --label2 $LB2 --use_corr $CORR
done
done