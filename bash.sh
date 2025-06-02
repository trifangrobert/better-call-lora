#!/bin/bash

echo "Starting experiments: $(date)"

for rank in 8 4 2; do
    echo "Running BRA with rank $rank: $(date)"
    python bra_lora.py --run_name tiny_bra_rank_$rank --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --rank $rank
    echo "Finished BRA rank $rank: $(date)"
done

for rank in 8 4 2; do
    echo "Running B_uniform with rank $rank: $(date)"
    python ba_init_lora.py --init_variant B_uniform --run_name tiny_b_uniform_a_zero_rank_$rank --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --rank $rank
    echo "Finished B_uniform rank $rank: $(date)"
done

for rank in 8 4 2; do
    echo "Running A_uniform with rank $rank: $(date)"
    python ba_init_lora.py --init_variant A_uniform --run_name tiny_b_zero_a_uniform_rank_$rank --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --rank $rank
    echo "Finished A_uniform rank $rank: $(date)"
done

echo "All done: $(date)"