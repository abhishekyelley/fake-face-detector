#!/bin/bash

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-opencv

# Install Python packages
pip install -r requirements.txt
