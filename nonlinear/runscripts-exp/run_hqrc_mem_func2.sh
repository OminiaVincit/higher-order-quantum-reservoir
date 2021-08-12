#!/usr/bin/bash
# Script to calculate memory function
export OMP_NUM_THREADS=1

BIN=../source/mf_hqrc.py
N=2
J=1.0
TASK=qrc_stm
SAVE=../../../data/hqrc/mem_func_batch_norm
QR=5
PROC=101
#SOLVER='ridge_pinv'
SOLVER='linear_pinv'

TAUS=\'-1,0,3\' # The parameters for tau is 2**x for x in TAUS
MIND=0
MAXD=200
INT=1
CB=0
SM=0
SG=1.0
TP=0
MASK=0

for a in 0.0 0.1 0.5
do
for V in 1
do
    python $BIN --type_input $TP --non_linear $SM --sigma_input $SG --mask_input $MASK --combine_input $CB  --solver $SOLVER --coupling $J --taudeltas $TAUS --taskname $TASK --nqrc $QR --strength $a --virtuals $V --ntrials $N --nproc $PROC --mind $MIND --maxd $MAXD --interval $INT --savedir $SAVE
done
done
