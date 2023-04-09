
#!/bin/bash

apt-get update && apt-get install libgl1

sudo apt-get update
sudo apt-get install -y python3-opencv

pip install -r requirements.txt
