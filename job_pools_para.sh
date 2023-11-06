#!/bin/bash

# Maximum number of parallel jobs
MAX_JOBS=$1

# Queue of Python programs with their respective arguments
QUEUE=(
    "python run_experiment_rl_delay.py  RoadRunner"
    "python run_experiment_rl_delay.py  StarGunner"
    "python run_experiment_rl_delay.py  TimePilot"
    "python run_experiment_rl_delay.py  Zaxxon"
    "python run_experiment_rl_delay.py  MsPacman"
    "python run_experiment_rl_delay.py  Qbert"
    "python run_experiment_rl_delay.py  NameThisGame"
    "python run_experiment_rl_delay.py  Pong"
    "python run_experiment_rl_delay.py  Freeway"
    # Add more programs and arguments as needed
)

# Function to run jobs in parallel
run_jobs() {
    local index=0
    local current_jobs=0

    while [ $index -lt ${#QUEUE[@]} ]; do
        # Run the job in the queue
        ${QUEUE[$index]} &

        # Increment the index and the number of current jobs
        ((index++))
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
