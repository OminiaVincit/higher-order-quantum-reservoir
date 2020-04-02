#!/usr/bin/bash
for v in 5 10 15 20 25 30 35 40 45 50
do
python source/mem_capacity.py --maxd 200 --ntrials 10 --virtuals $v 
done
