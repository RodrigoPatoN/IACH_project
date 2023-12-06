#!/bin/bash

# Check if jobs.txt file exists
if [ ! -f "jobs.txt" ]; then
    echo "jobs.txt not found."
    exit 1
fi

while true; do
    # Get the current number of words in jobs.txt
    num_lines=$(wc -w < jobs.txt)

    # Check if there are no more commands
    if [ "$num_lines" -eq 0 ]; then
        break
    fi

    # Read the first command from jobs.txt
    command=$(sed -n 1p jobs.txt)

    # Remove the executed command from jobs.txt
    sed -i '1d' jobs.txt

    # Check if the command is not empty
    if [ -n "$command" ]; then
        echo "Running command: $command"
        eval "$command"
    fi
done

echo "All commands have been executed."
