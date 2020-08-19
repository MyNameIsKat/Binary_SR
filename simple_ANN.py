
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras
import numpy as np
import pandas as pd

from keras.utils.vis_utils import plot_model
input_folder = 'final_dataset.csv'
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
import matplotlib.pyplot as plt


##### Use pandas to read the csv file into the dataset variable
dataset = pd.read_csv(input_folder)

##### save all 26-featured instances to the X variable
X = dataset.iloc[:, 0:26].values
##### save the ground truth vector to the y variable
y = dataset.iloc[:, 26:].values
# Reshape ground truth vectors to avoid error when split for validation
c, r = y.shape
y = y.reshape(c, )


ann_model = Sequential()
Input_layer = Dense(activation="relu", input_dim=26, units=300, kernel_initializer="random_normal",use_bias=True)
Hidden_layer2 = Dense(activation="relu", units=150, kernel_initializer="random_normal",use_bias=True)
Hidden_layer3 = Dense(activation="relu", units=150, kernel_initializer="random_normal",use_bias=True)
Output_layer = Dense(activation="sigmoid", units=1, kernel_initializer="random_normal",use_bias=True)

ann_model.add(Input_layer)
ann_model.add(Hidden_layer2)
ann_model.add(Hidden_layer3)
ann_model.add(Output_layer)

#keras function that helps output the structure and trainable parameters
print("------------------------------------------------------")
print("\nNEURAL NETWORK STRUCTURE AND TRAINABLE PARAMETERS")
ann_model.summary()
print("------------------------------------------------------")

#prints summary into an image
plot_model(ann_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
ann_model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ["accuracy"])

mod_loss = []
mod_vloss = []
mod_vacc = []

# Variables for confusion matrix
conf_mat_shape = (2,2)
conf_mat = np.zeros(conf_mat_shape)
conf_mat_all = np.zeros(conf_mat_shape)

print("\nApplying Stratified 10-Fold Validation. Early stopping implemented based on minimum training loss.")

# Create a stopping condition callback to stop epochs when training loss is minimized for that fold
stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='min')

# Create a random seed for use in validation initializing
seed = 7
np.random.seed(seed)

fivefold_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# Create loop for Stratified 5-fold cross validation and start training
count = 0

for train, validation in fivefold_strat.split(X, y):

	count = count + 1
	print("-------------------------------------------------------------------------------------------------")
	print("Fold " , count)

	history = ann_model.fit(X[train],y[train], epochs=110, batch_size=4, verbose=0,validation_data=(X[validation],y[validation]),callbacks=[stop])
	
	# Validation score per fold
	y_true = y[validation]
	y_pred = ann_model.predict(X[validation])
	confmat = metrics.confusion_matrix(y_true,(y_pred>0.5).astype(int))

	# Compute average loss, accuracy per fold by taking the mean across all epochs
	train_accuracy = np.asarray(history.history['accuracy']).mean()
	train_loss = np.asarray(history.history['loss']).mean()
	val_accuracy = np.asarray(history.history['val_accuracy']).mean()
	val_loss = np.asarray(history.history['val_loss']).mean()
	
	##### Append train_loss and val_loss to the variables you declared in step 6
	mod_loss.append(train_loss)
	mod_vloss.append(val_loss)
	mod_vacc.append(val_accuracy)

	
	conf_mat_all = np.add(conf_mat_all,confmat)

	# Print results of fold
	print("\nTraining Accuracy ==> ", train_accuracy , " | Training Loss ==> ", train_loss,
		"\nValidation Accuracy ==> ", val_accuracy, " | Validation Loss ==> ", val_loss)

#Declared variable mod_vacc for the model score
#mod_vacc = ann_model.evaluate(X[train], y[train], verbose=0)
print("-------------------------------------------------------------------------------------------------")
print("\nModel accuracy: ", np.mean(mod_vacc)*100, "(+/- ", np.std(mod_vacc)*100, ")")

# Print confusion matrix for all folds
print("\nConfusion Matrix after 5 folds: ")
TP = conf_mat_all[0][0]
TN = conf_mat_all[0][1]
FP = conf_mat_all[0][1]
FN = conf_mat_all[1][0]
label_list = ["Person 1", "Person 2"]
conf_list = np.array([[TP,TN],[FP,FN]])
print(pd.DataFrame(conf_list,label_list,label_list))


##### save function of keras to save the model as an h5 file with the naming convention <Surname1_Surname2_Surname3_Model>
ann_model.save("Cuyan_Gayao_Talon_Model.h5")
print("\nModel saved as an h5 file.")

##### Use matplotlib to show a visualization of the training and validation loss minimization, refer to the image in the student guide. Use the variables you have in step 9
plt.plot(mod_loss, 'b', label='Training Error')
plt.plot(mod_vloss, 'g', label='Validation Error')
plt.title('Model Loss')
plt.xlabel('5 Cross Fold')
plt.ylabel('Loss')
plt.xticks(np.arange(5))
plt.legend(['Training Error', 'Validation Error'], loc='upper right')
plt.show()