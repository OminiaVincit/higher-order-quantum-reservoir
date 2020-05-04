@set BIN=source\mem_capacity_highorder_tau_change.py
@set N=10
@set V=1
@set TASK=qrc_stm
@set SAVE=rescapa_high_stm2

python %BIN% --plot 1 --strength 0.0 --ntrials %N% --virtuals %V% --taskname %TASK% --layers 1 --nproc 21 --maxd 200 --savedir %SAVE%

@set L=2,3,4,5
python %BIN% --plot 1 --strength 0.0 --ntrials %N% --virtuals %V% --taskname %TASK% --layers %L% --nproc 21 --maxd 200 --savedir %SAVE%
