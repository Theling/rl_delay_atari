#!/bin/bash

# Function to launch a job
function launch_job {
    echo "Job $1 started: $(date)"
    # Your job commands go here
    sleep 5  # Example job, replace with your actual command
    echo "Job $1 finished: $(date)"
}

# Launch jobs one by one
launch_job 1
launch_job 2
launch_job 3

echo "All jobs completed."
