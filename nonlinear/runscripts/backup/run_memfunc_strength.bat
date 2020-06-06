@set BIN=source\mem_function_highorder_strength.py
@set N=1
@set MIND=0
@set MAXD=200
@set INT=50
@set NPROC=21
@set SDIR=res_highfunc_deep
@set V=1
@set NQR=5
@set DEEP=1

@set DT=0.25
python %BIN% --deep %DEEP% --taudelta %DT% --nqrc %NQR% --mind %MIND% --maxd %MAXD% --interval %INT% --nproc %NPROC% --ntrials %N% --virtuals %V% --savedir %SDIR%

@set DT=0.5
python %BIN% --deep %DEEP% --taudelta %DT% --nqrc %NQR% --mind %MIND% --maxd %MAXD% --interval %INT% --nproc %NPROC% --ntrials %N% --virtuals %V% --savedir %SDIR%

@set DT=1.0
python %BIN% --deep %DEEP% --taudelta %DT% --nqrc %NQR% --mind %MIND% --maxd %MAXD% --interval %INT% --nproc %NPROC% --ntrials %N% --virtuals %V% --savedir %SDIR%

@set DT=2.0
python %BIN% --deep %DEEP% --taudelta %DT% --nqrc %NQR% --mind %MIND% --maxd %MAXD% --interval %INT% --nproc %NPROC% --ntrials %N% --virtuals %V% --savedir %SDIR%

@set DT=4.0
python %BIN% --deep %DEEP% --taudelta %DT% --nqrc %NQR% --mind %MIND% --maxd %MAXD% --interval %INT% --nproc %NPROC% --ntrials %N% --virtuals %V% --savedir %SDIR%

@set DT=8.0
python %BIN% --deep %DEEP% --taudelta %DT% --nqrc %NQR% --mind %MIND% --maxd %MAXD% --interval %INT% --nproc %NPROC% --ntrials %N% --virtuals %V% --savedir %SDIR%

@set DT=16.0
python %BIN% --deep %DEEP% --taudelta %DT% --nqrc %NQR% --mind %MIND% --maxd %MAXD% --interval %INT% --nproc %NPROC% --ntrials %N% --virtuals %V% --savedir %SDIR%
