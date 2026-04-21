#!/bin/bash
set -euo pipefail

# Ported from assignment4-data/get_assets.sh.
# Original paths assumed the CS336 cluster layout (/data/classifiers) and
# the student module name (cs336_data/). In this unified repo we stage
# downloaded classifier binaries under assets/cs336_data/ by default, and
# still honor a pre-populated SOURCE_DIR (e.g. a cluster share) via env var.
SOURCE_DIR="${SOURCE_DIR:-/data/classifiers}"
ASSETS_DIR="${ASSETS_DIR:-$(pwd)/assets/cs336_data}"
mkdir -p "$ASSETS_DIR"

# Function to handle each file
handle_file() {
    local filename=$1
    local url=$2

    # Check if file exists in target directory
    if [ -e "$ASSETS_DIR/$filename" ]; then
        echo "File $filename already exists in $ASSETS_DIR, skipping..."
    # Check if file exists in source directory
    elif [ -f "$SOURCE_DIR/$filename" ]; then
        echo "File $filename exists in $SOURCE_DIR, creating softlink..."
        ln -s "$SOURCE_DIR/$filename" "$ASSETS_DIR/$filename"
    else
        echo "File $filename not found in $SOURCE_DIR, downloading..."
        wget "$url" -O "$ASSETS_DIR/$filename"
    fi
}

# Handle each file
handle_file "dolma_fasttext_nsfw_jigsaw_model.bin" "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin"
handle_file "dolma_fasttext_hatespeech_jigsaw_model.bin" "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin"