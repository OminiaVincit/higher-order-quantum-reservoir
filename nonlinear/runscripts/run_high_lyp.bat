@set BIN=source\lyapuv_highorder.py
@set N=2
@set S=1e-2
@set L=5
@set DELTA='-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7'
@set SDIR=res_high_lyp
@set BUFFER=1000
@set LENGTH=2000
@set V=1

@set ALPHA=0.1

@set J=0.125
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

@set ALPHA=0.5
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

@set ALPHA=0.9
python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

@set J=0.5
@rem python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

@set J=2.0
@rem python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

@set J=0.25
@rem python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

@set J=4.0
@rem python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

@set J=0.125
@rem python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

@set J=8.0
@rem python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

@set J=0.0625
@rem python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

@set J=16.0
@rem python %BIN% --buffer %BUFFER% --length %LENGTH%  --taudeltas %DELTA% --coupling %J% --savedir %SDIR% --strengths %ALPHA% --ntrials %N% --initial_distance %S% --virtuals %V% --layers %L%

