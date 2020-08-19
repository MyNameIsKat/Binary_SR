#This program extracts full 26 MFCCs for all .WAV files contained in the directories. 
#The parent folder consists of an arbitrary number of folders named after the word they contain. 
#Each word folder then contains 2 folders: Neutral or Not Neutral, which in turn contains the .WAV files of the utterances whose MFCCs are extracted.

import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc

if __name__=='__main__':

	#Save the input directory directory to input_folder
	input_folder = "./audio_dataset_pronunciation/"

	#Access the input directory
	for dirname in os.listdir(input_folder):

	    ##### uses the os module join method to get the name of each subfolder, append this to input_folder, and save to a variable
	    a_variable = os.path.join(input_folder, dirname)

	    ##### Complete the input to the os.listdir function below. 
	    for subname in os.listdir( a_variable ):

	    	# Get the name of the subsubfolder
	    	s_subfolder = os.path.join(a_variable + "/", subname)

	    	##### Define X to be a one dimensional numpy array below
	    	X = []

	    	# Iterate through all audio files and extract MFCCs 
	    	for filename in [x for x in os.listdir(s_subfolder) if x.endswith('.wav')]:
		        # Read the input file
		        filepath = os.path.join(s_subfolder, filename)
		        sampling_freq, audio = wavfile.read(filepath)

		        # Extract MFCC features. As per python_speech_features, features come in frames x numcep shape
		        print("Extracting MFCC for " + filename + "...")

		        mfcc_features = mfcc(audio, preemph=0.97,winlen=0.025, winstep=0.01, nfft=402, nfilt=26, appendEnergy=True, numcep=26, winfunc=np.hamming)

		        X = mfcc_features

		        #####the input parameter to a numpy function that exports individual csv files for each MFCC
		        np.savetxt(filename + "_mfcc_" + subname + ".csv", X, delimiter=",")

		        print("Completed.")
