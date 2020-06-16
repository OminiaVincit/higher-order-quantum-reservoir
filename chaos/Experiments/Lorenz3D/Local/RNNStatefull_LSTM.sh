#!/bin/bash
#export OMP_NUM_THREADS=12
cd ../../../Methods

for RDIM in 1
do
for SS in 100
do
for SL in 16
do
for KP in 1.0
do
    python3 RUN.py rnn_statefull \
    --mode all \
    --display_output 1 \
    --system_name Lorenz3D \
    --write_to_log 1 \
    --N 100000 \
    --N_used 1000 \
    --RDIM $RDIM \
    --noise_level 1 \
    --rnn_cell_type lstm \
    --unitary_cplex 1 \
    --unitary_capacity 2 \
    --reg 0.0 \
    --scaler standard \
    --initializer xavier \
    --sequence_length $SL \
    --dropout_keep_prob $KP \
    --zoneout_keep_prob $KP \
    --hidden_state_propagation_length 100 \
    --prediction_length 4 \
    --rnn_activation_str tanh \
    --rnn_num_layers 1 \
    --rnn_size_layers $SS \
    --subsample 1 \
    --batch_size 32 \
    --max_epochs 100 \
    --num_rounds 5 \
    --overfitting_patience 20 \
    --training_min_epochs 1 \
    --learning_rate 0.001 \
    --train_val_ratio 0.8 \
    --it_pred_length 500 \
    --n_tests 2 \
    --reference_train_time 1 \
    --buffer_train_time 0.2 \
    --retrain 0
done
done
done
done



