#!/bin/bash

# Go to project root (one directory up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Add a cron job for daily upload at 2 AM using absolute path to the upload script
CRON_JOB="0 2 * * * /usr/bin/python3 ${PROJECT_ROOT}/data_collector/upload_data_to_gcs.py >> ${PROJECT_ROOT}/data_collector/upload_data_to_gcs_log.txt  2>&1"

# Check if cron job already exists
crontab -l | grep -F "$CRON_JOB" > /dev/null

if [ $? -eq 0 ]; then
    echo "Cron job already exists. No changes made."
else
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "Cron job added."
fi
