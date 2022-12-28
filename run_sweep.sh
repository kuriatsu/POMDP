#!/bin/bash

python3 ras_value_iteration_sweep.py param_9_1.yaml &
python3 ras_value_iteration_sweep.py param_9_2.yaml &

python3 ras_value_iteration_sweep.py param_9_high_perf_1.yaml &
python3 ras_value_iteration_sweep.py param_9_high_perf_2.yaml &

python3 ras_value_iteration_sweep.py param_9_low_perf_1.yaml &
python3 ras_value_iteration_sweep.py param_9_low_perf_2.yaml &

python3 ras_value_iteration_sweep.py param_11_high_perf_1.yaml &
python3 ras_value_iteration_sweep.py param_11_high_perf_2.yaml &

