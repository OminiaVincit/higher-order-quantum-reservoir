@set BIN=source\esp_highorder.py
@set N=10
@set S=10
@set L=5
@set DELTA='-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7'
@set SDIR=res_high_echo4
@set BUFFER=9000
@set LENGTH=10000
@set V=1

@set ALPHA=0.0

@set J=0.5
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=2.0
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=0.25
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=4.0
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=0.125
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=8.0
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=0.0625
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=16.0
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=1.0
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set ALPHA=0.9

@set J=0.5
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=2.0
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=0.25
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=4.0
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=0.125
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=8.0
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=0.0625
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=16.0
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%

@set J=1.0
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --strials %S% --virtuals %V% --layers %L%
