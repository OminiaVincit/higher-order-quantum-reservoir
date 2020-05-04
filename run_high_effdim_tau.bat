@set BIN=source\eff_highorder_tau_change.py
@set N=10
@set V=15

python %BIN% --strength 0.0 --ntrials %N% --virtuals %V% --layers 1

@set L=2,3
python %BIN% --strength 0.0 --ntrials %N% --virtuals %V% --layers %L%
python %BIN% --strength 0.1 --ntrials %N% --virtuals %V% --layers %L%
python %BIN% --strength 0.3 --ntrials %N% --virtuals %V% --layers %L%
python %BIN% --strength 0.5 --ntrials %N% --virtuals %V% --layers %L%
python %BIN% --strength 0.7 --ntrials %N% --virtuals %V% --layers %L%
python %BIN% --strength 0.9 --ntrials %N% --virtuals %V% --layers %L%

@set L=4,5
python %BIN% --strength 0.0 --ntrials %N% --virtuals %V% --layers %L%
python %BIN% --strength 0.1 --ntrials %N% --virtuals %V% --layers %L%
python %BIN% --strength 0.3 --ntrials %N% --virtuals %V% --layers %L%
python %BIN% --strength 0.5 --ntrials %N% --virtuals %V% --layers %L%
python %BIN% --strength 0.7 --ntrials %N% --virtuals %V% --layers %L%
python %BIN% --strength 0.9 --ntrials %N% --virtuals %V% --layers %L%
