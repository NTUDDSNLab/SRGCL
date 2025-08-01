#!/bin/bash

# Description:
#   This script searches for and deletes all "processed" subdirectories
#   under a given top-level "dataset" directory (including subdirectories).
#
# Usage:
#   ./delete_processed_folders.sh <dataset_path> [-d]
#
# Arguments:
#   <dataset_path> : The root directory where the script starts searching.
#   -d             : (Optional) If specified, actually deletes the folders.
#                    Without this flag, the script only previews what will be deleted.

set -e

# --- Argument parsing ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_path> [-d]"
    exit 1
fi

DATASET_PATH="$1"
DELETE_MODE=false

if [ "$2" == "-d" ]; then
    DELETE_MODE=true
fi

echo "🔍 Scanning directory: $DATASET_PATH"
echo "🎯 Target subdirectories: */processed"

# --- Find all 'processed' directories under the dataset path ---
MATCHED_DIRS=$(find "$DATASET_PATH" -type d -name "processed")

if [ -z "$MATCHED_DIRS" ]; then
    echo "✅ No 'processed' directories found."
    exit 0
fi

echo "📋 Found the following 'processed' directories:"
echo "$MATCHED_DIRS"

# --- Dry-run or deletion ---
if [ "$DELETE_MODE" = true ]; then
    echo "⚠️ Deletion mode enabled. Deleting folders..."
    while IFS= read -r dir; do
        rm -rf "$dir"
        echo "🗑️ Deleted: $dir"
    done <<< "$MATCHED_DIRS"
    echo "✅ All 'processed' folders deleted."
else
    echo "💡 Dry-run mode. No folders were deleted."
    echo "    To actually delete the folders, rerun with the -d flag:"
    echo "    ./delete_processed_folders.sh \"$DATASET_PATH\" -d"
fi
