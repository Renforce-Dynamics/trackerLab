#!/bin/bash

CSV_DIR="datasets/LAFAN1_Retargeting_Dataset/g1"
OUTPUT_DIR="datasets/temp"
SCRIPT_PATH="./source/third_party/beyondMimic/scripts/data_replay/csv_to_npz.py"

mkdir -p "$OUTPUT_DIR"

for csv_file in "$CSV_DIR"/*.csv; do
    filename=$(basename "$csv_file" .csv)
    output_name="$OUTPUT_DIR/$filename"

    echo "Processing: $csv_file -> $output_name.npz"

    python "$SCRIPT_PATH" \
        --input_file "$csv_file" \
        --output_name "$output_name" \
        --headless
done

echo "All CSV files converted."
