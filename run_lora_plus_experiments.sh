#!/bin/bash

echo "Starting experiments: $(date)"

rank=64
alpha=16

lr_A_list=(1e-5 5e-5 1e-4)
lr_B_list=(1e-4 2e-4 4e-4)

for lr_A in "${lr_A_list[@]}"; do
    for lr_B in "${lr_B_list[@]}"; do
        if [[ "$lr_A" == "5e-5" && "$lr_B" == "2e-4" ]]; then
            echo "Skipping already completed experiment: lr_A=$lr_A, lr_B=$lr_B"
            continue
        fi
        run_name="loraplus_alpha_${alpha}_r_${rank}_lr_A_${lr_A}_lr_B_${lr_B}"
        echo "Running LoRA+ with lr_A=$lr_A, lr_B=$lr_B: $(date)"
        python lora_plus.py --init_variant A_uniform --run_name "$run_name" --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --rank $rank --alpha $alpha --lr_A $lr_A --lr_B $lr_B
        echo "Finished LoRA+ with lr_A=$lr_A, lr_B=$lr_B: $(date)"
    done
done

echo "All done: $(date)"