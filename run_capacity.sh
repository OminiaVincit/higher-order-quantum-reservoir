#!/usr/bin/bash
for v in 30 35 40 45 50
do
python source/mem_capacity.py --maxd 200 --ntrials 10 --virtuals $v 
done
