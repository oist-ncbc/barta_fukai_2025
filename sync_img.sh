#!/bin/bash

MANUSCRIPT_DIR="$HOME/manuscripts/StructuredInhibition/img"
SOURCE_DIR="img"

# Navigate to Overleaf project directory
cd "$MANUSCRIPT_DIR" || exit

# Copy new images
rsync -av --ignore-existing "$SOURCE_DIR/" "$MANUSCRIPT_DIR/"

# Add, commit, and push to Overleaf
git add .
git commit -m "Auto-sync images from SSH server"
git push origin master