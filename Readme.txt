Code is compatible with 3.8 >= PYTHON <= 3.11 (Since I'm using torch==2.1.1 and torchtext==0.16.1)
Dataset link : https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview

To Run, make sure you have required version of python installed. 

1) Run the setup.sh file to create a virtual env and install all the required pkgs. (Note this only install the cpu version of torch, gpu version require manual installation)

2) Run train.py script using "python train.py". Epochs and lr can also be configured using "python train.py --epochs 10 --lr 10"

3) When train.py runs, it asks for wandb authentication. Either do the offline by pressing 3, or use your wandb credential to access the live dashboard by using 1 or 2.

3) Run Inference.ipynb using Jupyter Notebook for inference.
