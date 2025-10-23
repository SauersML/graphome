#!/bin/bash
set -e

# AMY gene cluster sliding window analysis
# Region: chr1:103554644-103758692 (AMY2B to AMY1C)
# Window size: 5kb

GBZ="${1:-s3://human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.gbz}"
OUTPUT_DIR="${2:-amy_analysis_results}"
WINDOW_SIZE=2000

echo "=== AMY Gene Cluster NGEC Analysis ==="
echo "GBZ: $GBZ"
echo "Output directory: $OUTPUT_DIR"
echo "Window size: ${WINDOW_SIZE}bp"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define region (expanded 40kb in each direction from original 103554644-103758692)
START=103514644
END=103798692
REGION_LENGTH=$((END - START))

echo "Region: chr1:${START}-${END} (${REGION_LENGTH}bp)"
echo ""

# Generate windows
WINDOWS_FILE="$OUTPUT_DIR/windows.txt"
python3 << EOF
start = $START
end = $END
window_size = $WINDOW_SIZE

with open('$WINDOWS_FILE', 'w') as f:
    current = start
    while current + window_size <= end:
        f.write(f"{current} {current + window_size}\n")
        current += window_size
    # Add final partial window if needed
    if current < end:
        f.write(f"{current} {end}\n")

# Count windows
with open('$WINDOWS_FILE', 'r') as f:
    num_windows = len(f.readlines())
print(f"Generated {num_windows} windows")
EOF

NUM_WINDOWS=$(wc -l < "$WINDOWS_FILE")
echo "Total windows: $NUM_WINDOWS"
echo ""

# Run eigen-region for each window
RESULTS_FILE="$OUTPUT_DIR/ngec_results.txt"
> "$RESULTS_FILE"

echo "Running eigen-region analysis..."
start_time=$(date +%s)

window_num=1
while read start end; do
    echo -n "[$window_num/$NUM_WINDOWS] chr1:$start-$end ... "
    
    # Run eigen-region
    result=$(cargo run --release -- eigen-region \
        --gfa "$GBZ" \
        --region "chr1:$start-$end" 2>&1)
    
    # Extract metrics
    ngec=$(echo "$result" | grep "^NGEC:" | awk '{print $2}')
    nodes=$(echo "$result" | grep "^Nodes:" | awk '{print $2}')
    edges=$(echo "$result" | grep "^Edges:" | awk '{print $2}')
    
    # Handle empty regions
    if [ -z "$ngec" ]; then
        ngec="0.0"
        nodes="0"
        edges="0"
        echo "EMPTY"
    else
        echo "NGEC=$ngec (nodes=$nodes, edges=$edges)"
    fi
    
    # Save results: window_num start end ngec nodes edges
    echo "$window_num $start $end $ngec $nodes $edges" >> "$RESULTS_FILE"
    
    window_num=$((window_num + 1))
done < "$WINDOWS_FILE"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo ""
echo "Analysis complete in ${elapsed}s ($(($elapsed / 60))m $(($elapsed % 60))s)"
echo ""

# Generate plot
echo "Generating plot..."
PLOT_FILE="$OUTPUT_DIR/amy_ngec_plot.png"
python3 scripts/plot_amy_ngec.py "$RESULTS_FILE" "$PLOT_FILE"

echo ""
echo "=== Results ==="
echo "Data: $RESULTS_FILE"
echo "Plot: $PLOT_FILE"
echo ""
echo "Done!"
