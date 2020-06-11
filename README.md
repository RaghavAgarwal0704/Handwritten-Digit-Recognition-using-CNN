# Handwritten-Digit-Recognition-using-CNN

This CNN based project is used to recognize the handwritten digts.
Dataset: MNIST Digit Classification , 
Activation Functions: ReLu, Softmax ,
Optimizer: Adam ,
Accuracy: ~98.5%

Procedure to run the code:
1. Run the preprocess.py file, it will load the dataset and split it in training and testing data which will be saved in x_train.npy,y_train.npy,x_test.npy,y_test.npy
2. Now run the train.py file, it will build the model, the training data will be loaded and fitted in the model, the model will be saved in dig.json while the weights will be saved in dig.h5 file.
3. Run the test.py file, the model and the testing data will be loaded, the testing will be done and accuracy will be calculated. Save the predicted and true value of the labels.
4. Finally run the confusion_matrix.py which will plot the confusion matrix and save it in confusion_matrix.png



Feel free to contact me on agarwal.raghav0704@gmail.com :)
