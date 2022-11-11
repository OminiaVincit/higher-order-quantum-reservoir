#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runIPC_taus.py
BINPLOT=../postprocess/plot_IPC_tau.py
PARENT=../../../data/hqrc/IPC
#T=2000000
T=10000

TS=_T_$T
WD=50
VAR=4
DEG=4
#DELAYS='0,100,50,50,20,20,10,10'
#DELAYS='0,20,20,20,20,20,10,10'
DELAYS='0,100,50,50,20'
#
V='1'
NSPINS=5
NPROC=101
QR=1

THRES=0.0
DYNAMIC='full_random'
CAPA=5
WIDTH=0.1
CB=0
MASK=0

for QRIN in 4
do
for ALPHA in 0.0
do
FRS='XXX'
for SEED in 0
do
LBS=IPC_cb_$CB\_qrin_$QRIN\_seed_$SEED
SAVE=$PARENT\/$LBS
FRS=$FRS,$LBS

python $EXE  --combine_input $CB --alpha $ALPHA --mask_input $MASK --qr_input $QRIN --nqrc $QR --nproc $NPROC --spins $NSPINS --seed $SEED --dynamic $DYNAMIC --deg_delays $DELAYS --thres $THRES --virtuals $V --length $T --max_deg $DEG --max_window $WD --max_num_var $VAR --savedir $SAVE

P=mdeg_4_mvar_4
for THRES in 0.0 1e-4
do
python $BINPLOT --parent $PARENT --folders $FRS --T $T --thres $THRES --alpha $ALPHA --nqrc $QR --dynamic $DYNAMIC --virtuals $V --nspins $NSPINS --keystr $P  --max_capa $CAPA --width $WIDTH
done

done

done
done


