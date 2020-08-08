# import the necessary packages
import os
import cv2

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from azureml.core import Dataset, Run
import azureml.contrib.dataset
from azureml.contrib.dataset import FileHandlingOption, LabeledDatasetTask

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

from pyimagesearch.nn.conv import ShallowNet

# create the run context
run = Run.get_context()

# create the required folders for downloaded filed and outputs
os.makedirs('./download', exist_ok=True)
output_folder = './outputs'
os.makedirs(output_folder, exist_ok=True)

# get input dataset by name
labeled_dataset = run.input_datasets['animal_labels']

# download or moutn the data
animal_pd = labeled_dataset.to_pandas_dataframe(file_handling_option=FileHandlingOption.DOWNLOAD, target_path='./download/', overwrite_download=True)
animal_pd.to_csv('./outputs/animal_dataset.csv')

print("[INFO] loading images...")
# Load the labeles and imagepaths in a dataframe
labels = list(animal_pd['label'])
imagePaths = list(animal_pd['image_url'])

data = []

# loop over the input images for preprocessing
for (i, imagePath) in enumerate(imagePaths):
    
    #preprocess the image
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(32,32),interpolation=cv2.INTER_AREA)
    image = img_to_array(image)

    data.append(image)

(data, labels) = (np.array(data), np.array(labels))

#normalize images
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model

print("[INFO] Building and compiling model...")
inputShape = (32, 32, 3)
classes = 2
opt = SGD(lr=0.005)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",
    input_shape=inputShape))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(classes))
model.add(Activation("softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=32, epochs=50, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=[ "dogs", "panda"]))

# Log the metrics 
run.log_list("loss", H.history["loss"])
run.log_list("val_loss", H.history["val_loss"])
run.log_list("train_acc", H.history["accuracy"])
run.log_list("val_acc", H.history["val_accuracy"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 50), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
# Log the image
run.log_image('Training_Validation_Metrics', plot=plt)

# Save the trained model
model.save("./outputs/model.h5")