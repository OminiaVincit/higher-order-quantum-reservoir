#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

BIN=../source/mc_hqrc.py
N=2
J=1.0
TASK=qrc_stm
SAVE=../../../data/hqrc/memcapa_mask
PROC=101

TAUS=\'3\' # The parameters for tau is 2**x for x in TAUS
#STRENGTHS='0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.999,0.9999,0.99999'
STRENGTHS='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0'
VS='10'
QRS='5'

MIND=0
MAXD=200
INT=1
CB=0
SM=2
SG=10.0
TP=0

#for TAUS in '3'
for TAUS in \'-3\' \'-2\' \'-1\' \'0\' \'1\' \'2\' \'3\'
do
for MASK in 1
do
for SOLVER in 'linear_pinv'
do
    python $BIN --type_input $TP --non_linear $SM --sigma_input $SG --mask_input $MASK --combine_input $CB --solver $SOLVER --coupling $J --taudeltas $TAUS --taskname $TASK --layers $QRS --strengths $STRENGTHS --virtuals $VS --ntrials $N --nproc $PROC --mind $MIND --maxd $MAXD --interval $INT --savedir $SAVE
done
done
done