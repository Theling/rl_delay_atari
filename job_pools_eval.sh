# #!/bin/bash

# # Function to launch a job
# function launch_job {
#     echo "Job started: $(date)"
#     # Your job commands go here
#     python evaluate.py $1 # Example job, replace with your actual command
#     echo "Job finished: $(date)"
# }

# # Path to the directory containing subdirectories
# DIRECTORY_PATH="./logs_new/"

# # Iterate through directories and run the Python script
# for dir in $DIRECTORY_PATH*/; do
#     # Extract directory name without path
#     dir_name=$(basename $dir)
#     echo "$dir_name"
#     # Run the Python script and pass the directory name as an argument
#     launch_job "$dir_name"
# done


# echo "All jobs completed."


#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <max_parallel_jobs>"
    exit 1
fi

# Maximum number of parallel jobs
MAX_JOBS=$1

# Main directory containing subdirectories
MAIN_DIR="./logs_new2"

# Function to run jobs in parallel
run_jobs() {
    local current_jobs=0

    # Iterate over each subdirectory in the main directory
    for subdir in $MAIN_DIR/*/; do
        # Run the job for the current subdirectory
        dir_name=$(basename $subdir)
        echo "$dir_name"
        (python evaluate.py "$dir_name") &

        # Increment the number of current jobs
        ((current_jobs++))

        # Check if the maximum number of jobs is reached
        if [ $current_jobs -ge $MAX_JOBS ]; then
            # Wait for any background job to finish before starting a new one
            wait -n
            ((current_jobs--))
        fi
    done

    # Wait for all remaining background jobs to finish
    wait
}

# Run the jobs in parallel
run_jobs

echo "All jobs completed."
