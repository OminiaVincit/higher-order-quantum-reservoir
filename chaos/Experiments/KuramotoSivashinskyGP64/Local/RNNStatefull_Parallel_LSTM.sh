#!/bin/bash
#export OMP_NUM_THREADS=12

cd ../../../Methods
for AR in 80 100 120 150 500 1000
do
mpiexec -n 32 python3 RUN.py rnn_statefull_parallel \
--mode all \
--display_output 0 \
--system_name KuramotoSivashinskyGP64 \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--RDIM 64 \
--rnn_cell_type lstm \
--unitary_cplex 1 \
--unitary_capacity 2 \
--reg 0 \
--scaler standard \
--initializer xavier \
--sequence_length 4 \
--hidden_state_propagation_length 500 \
--prediction_length 4 \
--rnn_activation_str tanh \
--dropout_keep_prob 1.0 \
--zoneout_keep_prob 0.998 \
--rnn_num_layers 3 \
--rnn_size_layers $AR \
--subsample 2 \
--batch_size 32 \
--max_epochs 20 \
--num_rounds 5 \
--overfitting_patience 100 \
--training_min_epochs 1 \
--learning_rate 0.001 \
--train_val_ratio 0.8 \
--n_groups 32 \
--group_interaction_length 4 \
--it_pred_length 400 \
--n_tests 10 \
--reference_train_time 1 \
--buffer_train_time 0 \
--retrain 0 \
--noise_level 1
done



