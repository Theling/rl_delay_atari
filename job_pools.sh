#!/bin/bash

# Function to launch a job
function launch_job {
    echo "Job $1 started: $(date)"
    # Your job commands go here
    python run_experiment_rl_delay.py $2 # Example job, replace with your actual command
    echo "Job $1 finished: $(date)"
}

# Launch jobs one by one
launch_job 1 RoadRunner
launch_job 2 StarGunner
launch_job 3 TimePilot
launch_job 4 Zaxxon
launch_job 4 MsPacman
launch_job 5 Qbert
launch_job 6 NameThisGame
launch_job 7 Pong
launch_job 8 Freeway



echo "All jobs completed."