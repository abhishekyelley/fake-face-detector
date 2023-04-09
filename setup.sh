#!/bin/bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
sudo apt-get install -y python3-opencv
pip install -r requirements.txt
