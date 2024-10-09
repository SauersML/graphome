#!/bin/bash

start_node=0
end_node=200

while true; do
    output_file="multi_submatrix${start_node}_$((end_node)).gam"
    ./target/release/graphome extract --input adjacency_matrix.bin --start-node $start_node --end-node $end_node --output $output_file
    
    # Increment start-node and end-node by 5
    start_node=$((start_node + 5))
    end_node=$((end_node + 5))
done
