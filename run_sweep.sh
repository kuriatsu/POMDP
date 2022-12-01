#!/bin/bash

python3 ras_value_iteration_sweep.py param_2_1.yaml &
python3 ras_value_iteration_sweep.py param_2_2.yaml &
python3 ras_value_iteration_sweep.py param_2_3.yaml &
python3 ras_value_iteration_sweep.py param_2_4.yaml &
python3 ras_value_iteration_sweep.py param_2_5.yaml &
wait
python3 ras_value_iteration_sweep.py param_2_6.yaml &
python3 ras_value_iteration_sweep.py param_2_7.yaml &
python3 ras_value_iteration_sweep.py param_2_8.yaml &
python3 ras_value_iteration_sweep.py param_2_9.yaml &
python3 ras_value_iteration_sweep.py param_2_10.yaml &
wait
python3 ras_value_iteration_sweep.py param_2_11.yaml &
python3 ras_value_iteration_sweep.py param_2_12.yaml &
python3 ras_value_iteration_sweep.py param_2_13.yaml &
python3 ras_value_iteration_sweep.py param_2_14.yaml &
python3 ras_value_iteration_sweep.py param_2_15.yaml &
