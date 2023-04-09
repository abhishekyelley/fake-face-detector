
#!/bin/bash

sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libgl1
sudo ldconfig

sudo apt-get update
sudo apt-get install -y python3-opencv

pip install -r requirements.txt
