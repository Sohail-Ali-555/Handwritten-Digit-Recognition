# Python program for testing our ML classifier model(s)
# which will recognize and predict test digit values from MNIST dataset

# step 0: importing necessary modules
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
import matplotlib
from matplotlib import pyplot as plt
import random
import pickle
from tkinter import *
from PIL import Image, ImageTk

# step 1: fetch mnist data set, one of the latest versions from sklearn module itself
mnist = fetch_openml('mnist_784', parser="auto")

# step 2: Setting up Features (X) & Labels/Targets (Y)
#         involving making Numpy array of features (X)
X = mnist['data']
X_np = np.array(X)
Y = mnist['target']

# step 3: loading all the pre-trained models from models.pkl object
#         it is a bundle of 10 classifiers, with each acting as a binary classifier
#         for one of the ten digits, i.e., from 0-9
#         loading will only be done once, outside main whileloop
file_models = "models.pkl"
file_models_obj = open(file_models, 'rb')
models_list = pickle.load(file_models_obj)

# MAIN LOOP: (testing + GUI loop)
while True:
    # step 4: selecting a random data point from testing part, from last 10000 rows
    random_number = random.randint(60000, 70000)

    # test features
    random_digit = X_np[random_number]
    # test label
    random_digit_label = Y[random_number]

    # step 5: reshaping the 1 x 784 pixels / features array into
    #         a pixelated image equivalent; i.e. 28 x 28 2-D pixels array

    random_digit_image = random_digit.reshape(28, 28)

    # step 6: plotting the digit graphically using Matplotlib
    matplotlib.use('Agg')

    plt.imshow(random_digit_image, cmap=plt.cm.binary, interpolation='nearest')

    # axis can be turned on / off
    plt.axis("on")

    # we will first save the image as '.png' and then embed it in our GUI format
    # with every test loop a new image will replace the old one
    plt.savefig('random_image.png')

    """ 
    Beginning of ML testing / prediction session
    """

    # initial test digit label taken as None / Null
    digit_label = None

    # step 7: loop in range (0, 10) to perform Binary Classification for each
    #         of the 10 digits
    for i in range(10):

        # step 8: doing prediction / test operation for the random test digit
        result = models_list[i].predict([random_digit]).astype(bool)

        # if received a resultant label, i.e. True condition
        # it will be taken as a value from 0-9, and loop will be broken
        if result:
            digit_label = i
            break
    # if resultant label obtained
    else:
        print("Error occurred during Training / Prediction")

    # step 9: GUI work; we will show program output using a GUI format
    root = Tk()
    root.geometry('760x620')
    root.title("MNIST - ML Project")

    Label(root, text='\nMNIST DataSet Hand Written Digit Recognition', font='consolas 20 bold underline').pack()

    digit_photo = Image.open('random_image.png')

    # embed the digit image in GUI window
    if digit_photo is None:
        Label(root, text=f'\nCannot Load Image | Error Occurred', font='15').pack()
    else:
        resized_image = digit_photo.resize((400, 300), Image.LANCZOS)

        resized_image_Tk = ImageTk.PhotoImage(resized_image)

        photo_Label = Label(root, image=resized_image_Tk)

        photo_Label.pack()

    # step 10: giving output / conclusion of Testing

    if digit_label is None:
        Label(root, text=f'\nPredicted Graphical Digit is :- {digit_label} | Training Error', font='15').pack()
    else:
        Label(root, text=f'\nPredicted Graphical Digit is :- {digit_label}', font='12').pack()

    Label(root, text=f'\n    Actual Graphical Digit is :- {random_digit_label}', font='12').pack()

    if digit_label == int(random_digit_label):
        Label(root, text=f'\nResult : Correct Prediction', font='12').pack()
    else:
        Label(root, text=f'\nResult : Wrong Prediction', font='12').pack()

    # step 11: exiting the program : y / n ?, using exit() function

    def exit_prog():
        print("Exited the Program")
        exit()


    # completing the GUI part, by adding the exit() button, normal exit will make the
    # program to continue running in infinite loop
    Label(text='\n').pack()
    Button(root, text=" >> Exit <<", font='12', command=exit_prog).pack()

    root.mainloop()
