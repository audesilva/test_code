# Disclaimer:
Rip current detection is an experimental product and should not be used for decisions regarding public safety. It is presently being evaluated at different locations and under different oceanographic and environmental conditions. There are known limitations with the existing model, such as not recognizing all types of rip currents or not detecting rip currents with unclear visual indicators. The model should presently only be used for research purposes.

a.  https://www.sciencedirect.com/science/article/abs/pii/S0378383921000193
    
b.  general info about rips e.g.
     https://www.sciencedirect.com/science/article/pii/S0012825216303117


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
