#!/bin/bash

echo "Starting experiments: $(date)"

init_variant=pissa
rank_list=(2 4 8)
lr=5e-5

for rank in "${rank_list[@]}"; do
    run_name="pissa_init_${init_variant}_r_${rank}_alpha_${rank}_lr_${lr}"
    echo "Running PiSSA with rank=$rank, alpha=$rank: $(date)"
    python .\\pissa.py --init_variant $init_variant --rank $rank --alpha $rank --lr $lr --run_name $run_name
    echo "Finished PiSSA with rank=$rank, alpha=$rank: $(date)\n"
done

echo "All done: $(date)"
