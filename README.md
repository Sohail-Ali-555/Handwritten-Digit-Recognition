# Handwritten-Digit-Recognition
A Python program that recognizes and classifies a handwritten digit into one of the ten possible digits from 0-9.
## Installing pre-requisite Python libraries
To install the modules, download the [**requirements.txt**](https://github.com/Sohail-Ali-555/Handwritten-Digit-Recognition/blob/main/requirements.txt) file and open command prompt or powershell window in the same directory and type-in the following command:
**pip install -r requirements.txt**

Note: Modules like NumPy, Scikit-Learn, Matplotlib, Pillow are chosen for their latest versions as of this commit date, since latest versions will generally offer max. possible performance speed.

## Pre-trained Models.pkl file
A [**models.pkl**](https://github.com/Sohail-Ali-555/Handwritten-Digit-Recognition/blob/main/models.pkl) file is present in the reposiroty which has been obtained on running the [**training.py**](https://github.com/Sohail-Ali-555/Handwritten-Digit-Recognition/blob/main/training.py) program file. You can also obtain it by running the same program file or using this one.

## MNIST dataset
Due to ease of programming we have used the MNIST dataset provided within the **Sckit-Learn** module itself. But you can download and use the MNIST dataset from [official sources](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) too. 

## Running the program
Their are two python files namely, [**training.py**](https://github.com/Sohail-Ali-555/Handwritten-Digit-Recognition/blob/main/training.py) & [**testing.py**](https://github.com/Sohail-Ali-555/Handwritten-Digit-Recognition/blob/main/testing.py).

### Training Session
Training file will take considerably more time to be executed completely and will generate a **models.pkl**. This basically comprises the learning session of our ML model, where we are producing 10 binary classifiers, each capable of detecting only one out of 10 digits (0-9), giving result in boolean form. The ten models have been packed together as one **models.pkl** binary-readable only object file.
### Testing Session
Testing file will be executed much faster, which involves first unpacking **models.pkl** file into a python list of binary classifiers and pitting them against a randomly selected digit from the **MNIST** dataset for prediction. 
