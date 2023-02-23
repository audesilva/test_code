# Rip Current Detector Beta Testing Code


## 1. Download weights of the model

mkdir models <br>
wget https://www.dropbox.com/s/dcsdi36jbc570u9/fasterrcnn_resnet50_fpn.pt -O models/fasterrcnn_resnet50_fpn.pt

## 2. Clone this repo

git clone https://github.com/audesilva/test_code.git

## 3. Install the requirements

pip install -r ./test_code/requirements.txt

## 4. Predict on the video stream

python ./test_code/main.py 'currituck_hampton_inn'
