@set BIN=source\narma_nmse_highorder_strength_change.py
@set N=1
@set V=20
@set DT=1
@set SDIR=resnarma_high_strength_deep
@set STR=0.2,0.4,0.6,0.8,1.0
@set DEEP=1
@set LAYERS=2,3,4,5

python %BIN% --deep %DEEP% --ntrials %N% --virtuals %V% --taudelta %DT% --layers %LAYERS% --orders 20 --savedir %SDIR% --strengths %STR%
python %BIN% --deep %DEEP% --ntrials %N% --virtuals %V% --taudelta %DT% --layers %LAYERS% --orders 15 --savedir %SDIR% --strengths %STR%
python %BIN% --deep %DEEP% --ntrials %N% --virtuals %V% --taudelta %DT% --layers %LAYERS% --orders 10 --savedir %SDIR% --strengths %STR%
python %BIN% --deep %DEEP% --ntrials %N% --virtuals %V% --taudelta %DT% --layers %LAYERS% --orders 5  --savedir %SDIR% --strengths %STR%

