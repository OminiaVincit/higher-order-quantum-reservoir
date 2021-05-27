#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1
MNIST_SIZE="10x10"

BIN=../source/mnist_hqrc.py
SAVE=../../../data/hqrc/mnist
#STRENGTHS='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9'
#DS='0.0,0.001,0.01,0.1,1.0,10.0,100.0'
DS='0.0'
FULL=1
LB1=3
LB2=5
CORR=1
LN=0
TRANS=100
N=10
DYNAMIC='phase_trans'
#'full_random,half_random,full_const_trans,full_const_coeff,ion_trap,phase_trans'
STRENGTHS='0.5'
TAUS=\'-5,-4,-3,-2,-1,0,1,2,3,4,5,6\'
WIDTH=10

for RATE in 0.1
do
for NQR in 10
do
for V in 10 20
do
python $BIN --taudeltas $TAUS --width $WIDTH --dynamic $DYNAMIC --non_diags $DS --ntrials $N --transient $TRANS --rate $RATE --linear_reg $LN --mnist_size $MNIST_SIZE --savedir $SAVE --nqrs $NQR --virtuals $V --full $FULL --label1 $LB1 --label2 $LB2 --strengths $STRENGTHS --use_corr $CORR
done
done
done
