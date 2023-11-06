#!/bin/bash

# Function to launch a job
function launch_job {
    echo "Job started: $(date)"
    # Your job commands go here
    python evaluate.py $1 # Example job, replace with your actual command
    echo "Job finished: $(date)"
}

# Path to the directory containing subdirectories
DIRECTORY_PATH="./logs/"

# Iterate through directories and run the Python script
for dir in $DIRECTORY_PATH*/; do
    # Extract directory name without path
    dir_name=$(basename $dir)
    echo "$dir_name"
    # Run the Python script and pass the directory name as an argument
    launch_job "$dir_name"
done


echo "All jobs completed."