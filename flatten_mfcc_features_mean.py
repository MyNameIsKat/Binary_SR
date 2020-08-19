#This program takes in .csv files from mfcc_feature_extractor and collapses the number of frames by getting the mean of each mfcc per frame per audio file
#This will be used as the word's corresponding features to be inputted into a model. 
#The program also appends the ground truth for each audio file.
#This program's output is a .csv file given by word x mfccs + a last column for the ground truth

import os
import regex as re
import pandas as pd
import numpy as np

if __name__=='__main__':

	#Initialize variables
	mean_mfcc = np.array([])
	mfcc_all = np.array([])
	ground_truth = np.array([])
	labels = np.array([])
	outfile = "final_dataset.csv"
	input_folder = "./flatten_mfcc/"
	file_counter = 0

	# Parse the input directory
	for file in os.listdir(input_folder):
		mean_mfcc = np.empty((0,26)) #clear array after each file

		#Get ground truth of file and append to array for later use
		gt_exp = re.search(r"Person1\b", str(file))

		if gt_exp != None:
			ground_truth = np.append(ground_truth, 1)
		else:
			ground_truth = np.append(ground_truth, 0)

		# Save instance names for later use
		labels = np.append(labels, str(file))

		print("Computing mean MFCC for " + str(file) + "...")	

		for filename in [x for x in os.listdir(input_folder) if x.endswith('.csv')]:
			#counter for each column
			counter = 0

			# Read the input file into a pandas data frame
			filepath = os.path.join(input_folder, filename)
			df = pd.read_csv(filepath)

			#Compute the mean of each column and append to an array
			for cols in df.columns:
				mean_mfcc = np.append(mean_mfcc, df.iloc[:, counter].mean())
				counter = counter + 1

		file_counter = file_counter + 1

		print("Done.")

	# Append mean_mfccs of all files into 1 array
	mfcc_all = np.append(mfcc_all, mean_mfcc, axis=0)

	# Reshape array to be array of arrays separated per file x 26 mfccs
	mfcc_all = mfcc_all.reshape(file_counter, 26)	

	# Reshape ground truth and append to dataset
	ground_truth = ground_truth.reshape(-1,1)
	mfcc_all = np.append(mfcc_all, ground_truth, axis=1)

	# Reshape instance labels
	labels = labels.reshape(-1,1)

	#Converge into 1 dataset
	instances_df = pd.DataFrame(labels, columns=['File']) 
	mfccs_df = pd.DataFrame(mfcc_all, columns=['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13','F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'Ground Truth'])
	dataset_df = pd.concat([instances_df, mfccs_df], axis=1)

	# Convert data frame into a csv
	print("\nSaving dataset for ", file_counter , " audio samples.")
	dataset_df.to_csv(outfile, encoding='utf-8', index=False)

	print("\nDone.")




	


		
		
			

