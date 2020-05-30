@set BIN=source\mem_function_highorder_tau.py
@set N=10
@set MIND=0
@set MAXD=50
@set NPROC=20
@set SDIR=rescapa_highfunc_deep
@set V=1

@set STR=0.9

python %BIN% --deep 1 --nqrc 3 --mind %MIND% --maxd %MAXD% --nproc %NPROC% --ntrials %N% --virtuals %V% --strength %STR% --savedir %SDIR%
python %BIN% --deep 1 --nqrc 5 --mind %MIND% --maxd %MAXD% --nproc %NPROC% --ntrials %N% --virtuals %V% --strength %STR% --savedir %SDIR%

@set STR=0.5

python %BIN% --deep 1 --nqrc 3 --mind %MIND% --maxd %MAXD% --nproc %NPROC% --ntrials %N% --virtuals %V% --strength %STR% --savedir %SDIR%
python %BIN% --deep 1 --nqrc 5 --mind %MIND% --maxd %MAXD% --nproc %NPROC% --ntrials %N% --virtuals %V% --strength %STR% --savedir %SDIR%
