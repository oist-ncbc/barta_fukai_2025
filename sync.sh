#!/bin/bash

# Define project path (assuming the script is run from the project folder)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$PROJECT_DIR/config.yaml"

# Function to extract values from YAML
extract_yaml_value() {
    grep "$1" "$CONFIG_FILE" | awk -F': ' '{print $2}'
}

# Read paths from config.yaml
LOCAL_DATA_DIR=$(extract_yaml_value "data_path")
REMOTE_USER=$(extract_yaml_value "remote_user")
REMOTE_HOST=$(extract_yaml_value "remote_host")
REMOTE_DATA_DIR=$(extract_yaml_value "remote_data_path")

# Ensure the variables are not empty
if [[ -z "$LOCAL_DATA_DIR" || -z "$REMOTE_USER" || -z "$REMOTE_HOST" || -z "$REMOTE_DATA_DIR" ]]; then
    echo "❌ Error: Missing required values in config.yaml"
    exit 1
fi

# Ensure local data directory exists
if [[ ! -d "$LOCAL_DATA_DIR" ]]; then
    echo "❌ Error: Local data directory does not exist: $LOCAL_DATA_DIR"
    exit 1
fi

echo "🔄 Syncing data..."
echo "📂 Local Path: $LOCAL_DATA_DIR"
echo "🌍 Remote Path: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DATA_DIR"

# Check if local data is newer than remote data
if [ "$(find "$LOCAL_DATA_DIR" -type f -newermt '1 minute ago' | wc -l)" -gt 0 ]; then
    echo "📤 Syncing local data to remote..."
    rsync -avz --progress "$LOCAL_DATA_DIR" "${REMOTE_USER}@${REMOTE_HOST}:$REMOTE_DATA_DIR"
else
    echo "📥 Syncing remote data to local..."
    rsync -avz --progress "${REMOTE_USER}@${REMOTE_HOST}:$REMOTE_DATA_DIR" "$LOCAL_DATA_DIR"
fi

echo "✅ Sync complete!"
