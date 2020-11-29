# src/train.py
import jolib
import os
import argparse
import pandas as pandas
from sklearn import metrics
from sklearn import tree

import config
import model_dispatcher

def run(fold, model):	
	# read training data with fold
	df = pd.read_csv(config.TRAINING_FILE)

	# training data is where fold is not equal to provided fold
	# also we reset the index
	df_train = df[df.fold != fold].reset_index(drop=True)

	# calidation data is where fold is equal to provided fold
	df_valid = df[df.fold == fold].reset_index(drop=True)

	# drop the label column from dataframe and convert
	# to numpy array using .values
	# target is label column in the dataframe
	x_train = df_train.drop("label", axis=1)
	y_train = df_train.label.values

	# similarly for validations
	x_valid = df_valid.drop("label", axis=1)
	y_train = df_valid.label.values

	# initialize simple decision tree
	clf = model_dispatcher.models(model)

	# fit the model
	clf.fit(x_train, y_train)

	# create predictions for validation data
	preds = clf.predict(x_valid)

	# accruacy
	accuracy = metrics.accuracy_score(y_valid, preds)
	print(f"fold = {fold} accuracy = {accuracy}")

	# save the model
	jolib.dump(
		clf,
		os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
		)


if __name__ == "__main__":
	# initialize ArgumentParser class of argparse
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--folds",
		type=int
	)

	parser.add_argument(
		"--model",
		type=str
	)

	# add different arguments you need and their type
	args = parser.parse_args()

	# run fold specified by commandline arguments (ie "python train.py --fold 0 --model decision_tree_gini" 
	# or run shell script "sh run.sh")
	run(fold=args.fold, model=args.model)