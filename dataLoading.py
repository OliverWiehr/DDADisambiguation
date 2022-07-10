import os
import numpy as np
from sklearn.model_selection import train_test_split

STANDARD_DATA_TYPE = 'concat'

def loadCTDDDAs(dataType=STANDARD_DATA_TYPE):
	X_pos = np.array([np.load(f"data/CTDDDAs_{dataType}/therapeutic/{filename}") for filename in os.listdir(f"data/CTDDDAs_{dataType}/therapeutic")])
	X_neg = np.array([np.load(f"data/CTDDDAs_{dataType}/marker/{filename}") for filename in os.listdir(f"data/CTDDDAs_{dataType}/marker")])

	return X_pos, X_neg

def loadCTDDDAsWithSemantics(dataType='sparse'):
	X_pos = np.array([np.load(f"data/CTDDDAsWithSemantics_{dataType}/therapeutic/{filename}") \
		for filename in os.listdir(f"data/CTDDDAsWithSemantics_{dataType}/therapeutic")])
	X_neg = np.array([np.load(f"data/CTDDDAsWithSemantics_{dataType}/marker/{filename}") \
		for filename in os.listdir(f"data/CTDDDAsWithSemantics_{dataType}/marker")])

	return X_pos, X_neg
	
def loadPredictedDDAs(NDNs, dataType=STANDARD_DATA_TYPE):
	X_pos = np.array([np.load(f"data/predictedDDAs-{NDNs}NDNs_{dataType}/therapeutic/{filename}") \
				for filename in os.listdir(f"data/predictedDDAs-{NDNs}NDNs_{dataType}/therapeutic")])
	X_neg = np.array([np.load(f"data/predictedDDAs-{NDNs}NDNs_{dataType}/marker/{filename}") \
				for filename in os.listdir(f"data/predictedDDAs-{NDNs}NDNs_{dataType}/marker")])

	return X_pos, X_neg

def loadPredictedDDAsWithSemantics(NDNs, dataType, numSamples=None):
	pos_ids = np.random.permutation(os.listdir(f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/therapeutic"))
	neg_ids = np.random.permutation(os.listdir(f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/marker"))
	
	if numSamples is None:
		pos_train_ids = pos_ids[:int(len(pos_ids) * 0.9)]
		pos_test_ids = pos_ids[int(len(pos_ids) * 0.9):]
		neg_train_ids = neg_ids[:int(len(neg_ids) * 0.9)]
		neg_test_ids = neg_ids[int(len(neg_ids) * 0.9):]
	else:
		pos_train_ids = pos_ids[:int(numSamples * 0.9)]
		pos_test_ids = pos_ids[int(numSamples * 0.9):]
		neg_train_ids = neg_ids[:int(numSamples * 0.9)]
		neg_test_ids = neg_ids[int(numSamples * 0.9):]
	
	X_pos_train_directories = [f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/therapeutic/{dda_id}" for dda_id in pos_train_ids]
	X_neg_train_directories = [f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/marker/{dda_id}" for dda_id in neg_train_ids]
	X_pos_test_directories = [f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/therapeutic/{dda_id}" for dda_id in pos_test_ids]
	X_neg_test_directories = [f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/marker/{dda_id}" for dda_id in neg_test_ids]
	
	X_pos_train_files = []
	for directory in X_pos_train_directories:
		X_pos_train_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]
	X_neg_train_files = []
	for directory in X_neg_train_directories:
		X_neg_train_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]
	X_pos_test_files = []
	for directory in X_pos_test_directories:
		X_pos_test_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]
	X_neg_test_files = []
	for directory in X_neg_test_directories:
		X_neg_test_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]

	if len(X_pos_train_files) > len(X_neg_train_files):
		X_pos_train_files = X_pos_train_files[:len(X_neg_train_files)]
	elif len(X_neg_train_files) > len(X_pos_train_files):
		X_neg_train_files = X_neg_train_files[:len(X_pos_train_files)]
	if len(X_pos_test_files) > len(X_neg_test_files):
		X_pos_test_files = X_pos_test_files[:len(X_neg_test_files)]
	elif len(X_neg_test_files) > len(X_pos_test_files):
		X_neg_test_files = X_neg_test_files[:len(X_pos_test_files)]

	if len(X_pos_train_files) > len(X_pos_test_files) * 9:
		X_pos_train_files = X_pos_train_files[:len(X_pos_test_files) * 9]
		X_neg_train_files = X_neg_train_files[:len(X_neg_test_files) * 9]
	elif len(X_pos_test_files) > int(len(X_pos_train_files) / 9):
		X_pos_test_files = X_pos_test_files[:int(len(X_pos_train_files) / 9)]
		X_neg_test_files = X_neg_test_files[:int(len(X_neg_train_files) / 9)]

	X_pos_train = np.array([np.load(file) for file in X_pos_train_files])
	X_neg_train = np.array([np.load(file) for file in X_neg_train_files])
	X_pos_test = np.array([np.load(file) for file in X_pos_test_files])
	X_neg_test = np.array([np.load(file) for file in X_neg_test_files])

	return X_pos_train, X_neg_train, X_pos_test, X_neg_test
	
def loadExistingDDAs(minCount, dataType=STANDARD_DATA_TYPE):
	X_pos = np.array([np.load(f"data/existingDDAs-{2019}-minCount{minCount}_{dataType}/therapeutic/{filename}") \
				for filename in os.listdir(f"data/existingDDAs-{2019}-minCount{minCount}_{dataType}/therapeutic")])
	X_neg = np.array([np.load(f"data/existingDDAs-{2019}-minCount{minCount}_{dataType}/marker/{filename}") \
				for filename in os.listdir(f"data/existingDDAs-{2019}-minCount{minCount}_{dataType}/marker")])

	return X_pos, X_neg

def loadRetrospectiveAnalysisData(NDNs, dataType=STANDARD_DATA_TYPE):
	X_pos_train = np.array([np.load(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}/train/therapeutic/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}/train/therapeutic")])
	X_neg_train = np.array([np.load(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}/train/marker/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}/train/marker")])
	X_pos_test = np.array([np.load(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}/test/therapeutic/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}/test/therapeutic")])
	X_neg_test = np.array([np.load(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}/test/marker/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}/test/marker")])

	return X_pos_train, X_neg_train, X_pos_test, X_neg_test

def loadRetrospectiveAnalysisDataWithDistance(NDNs):
	X_pos_train = np.array([np.load(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/train/therapeutic/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/train/therapeutic")])
	X_neg_train = np.array([np.load(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/train/marker/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/train/marker")])
	X_pos_test = np.array([np.load(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/test/therapeutic/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/test/therapeutic")])
	X_neg_test = np.array([np.load(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/test/marker/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/test/marker")])

	return X_pos_train, X_neg_train, X_pos_test, X_neg_test

def loadRetrospectiveAnalysisDataWithSemantics(NDNs, dataType='sparse'):
	X_pos_train = np.array([np.load(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/therapeutic/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/therapeutic")])
	X_neg_train = np.array([np.load(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/marker/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/marker")])
	X_pos_test = np.array([np.load(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/therapeutic/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/therapeutic")])
	X_neg_test = np.array([np.load(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/marker/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/marker")])

	return X_pos_train, X_neg_train, X_pos_test, X_neg_test

def loadRetrospectiveAnalysisDatasetWithSemantics(NDNs, dataType='full'):
	X_pos_train = np.array([np.load(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/therapeutic/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/therapeutic")])
	X_neg_train = np.array([np.load(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/marker/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/marker")])
	X_pos_test = np.array([np.load(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/therapeutic/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/therapeutic")])
	X_neg_test = np.array([np.load(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/marker/{filename}") \
				for filename in os.listdir(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/marker")])

	return X_pos_train, X_neg_train, X_pos_test, X_neg_test

def loadWeightedRetrospectiveAnalysisData(NDNs, dataType=STANDARD_DATA_TYPE):
	X_pos_train = np.array([np.load(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/train/therapeutic/{filename}") \
				for filename in os.listdir(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/train/therapeutic")])
	X_neg_train = np.array([np.load(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/train/marker/{filename}") \
				for filename in os.listdir(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/train/marker")])
	X_pos_test = np.array([np.load(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/test/therapeutic/{filename}") \
				for filename in os.listdir(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/test/therapeutic")])
	X_neg_test = np.array([np.load(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/test/marker/{filename}") \
				for filename in os.listdir(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/test/marker")])

	return X_pos_train, X_neg_train, X_pos_test, X_neg_test

def loadSemanticCTDDDAs():
	pos_ids = np.random.permutation(os.listdir(f"data/semanticCTDDDAs/therapeutic"))
	neg_ids = np.random.permutation(os.listdir(f"data/semanticCTDDDAs/marker"))

	pos_train_ids = pos_ids[:int(len(pos_ids) * 0.9)]
	pos_test_ids = pos_ids[int(len(pos_ids) * 0.9):]
	neg_train_ids = neg_ids[:int(len(neg_ids) * 0.9)]
	neg_test_ids = neg_ids[int(len(neg_ids) * 0.9):]

	X_pos_train_directories = [f"data/semanticCTDDDAs/therapeutic/{dda_id}" for dda_id in pos_train_ids]
	X_neg_train_directories = [f"data/semanticCTDDDAs/marker/{dda_id}" for dda_id in neg_train_ids]
	X_pos_test_directories = [f"data/semanticCTDDDAs/therapeutic/{dda_id}" for dda_id in pos_test_ids]
	X_neg_test_directories = [f"data/semanticCTDDDAs/marker/{dda_id}" for dda_id in neg_test_ids]

	X_pos_train_files = []
	for directory in X_pos_train_directories:
		X_pos_train_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]
	X_neg_train_files = []
	for directory in X_pos_train_directories:
		X_neg_train_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]
	X_pos_test_files = []
	for directory in X_pos_train_directories:
		X_pos_test_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]
	X_neg_test_files = []
	for directory in X_pos_train_directories:
		X_neg_test_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]

	X_pos_train = np.array([np.load(file) for file in X_pos_train_files])
	X_neg_train = np.array([np.load(file) for file in X_neg_train_files])
	X_pos_test = np.array([np.load(file) for file in X_pos_test_files])
	X_neg_test = np.array([np.load(file) for file in X_neg_test_files])

	return X_pos_train, X_neg_train, X_pos_test, X_neg_test

def loadSemanticCTDDDAsWithPubmed():
	pos_ids = np.random.permutation(os.listdir(f"data/semanticCTDDDAsWithPubmed/therapeutic"))
	neg_ids = np.random.permutation(os.listdir(f"data/semanticCTDDDAsWithPubmed/marker"))

	pos_train_ids = pos_ids[:int(len(pos_ids) * 0.9)]
	pos_test_ids = pos_ids[int(len(pos_ids) * 0.9):]
	neg_train_ids = neg_ids[:int(len(neg_ids) * 0.9)]
	neg_test_ids = neg_ids[int(len(neg_ids) * 0.9):]

	X_pos_train_directories = [f"data/semanticCTDDDAsWithPubmed/therapeutic/{dda_id}" for dda_id in pos_train_ids]
	X_neg_train_directories = [f"data/semanticCTDDDAsWithPubmed/marker/{dda_id}" for dda_id in neg_train_ids]
	X_pos_test_directories = [f"data/semanticCTDDDAsWithPubmed/therapeutic/{dda_id}" for dda_id in pos_test_ids]
	X_neg_test_directories = [f"data/semanticCTDDDAsWithPubmed/marker/{dda_id}" for dda_id in neg_test_ids]

	X_pos_train_files = []
	for directory in X_pos_train_directories:
		X_pos_train_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]
	X_neg_train_files = []
	for directory in X_neg_train_directories:
		X_neg_train_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]
	X_pos_test_files = []
	for directory in X_pos_test_directories:
		X_pos_test_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]
	X_neg_test_files = []
	for directory in X_neg_test_directories:
		X_neg_test_files += [f"{directory}/{filename}" for filename in os.listdir(directory)]

	X_pos_train = np.array([np.load(file) for file in X_pos_train_files])
	X_neg_train = np.array([np.load(file) for file in X_neg_train_files])
	X_pos_test = np.array([np.load(file) for file in X_pos_test_files])
	X_neg_test = np.array([np.load(file) for file in X_neg_test_files])

	return X_pos_train, X_neg_train, X_pos_test, X_neg_test

def loadSemanticRetrospectiveAnalysisData(NDNs):
	X_pos_train = np.array([np.load(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/train/therapeutic/{filename}") \
				for filename in os.listdir(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/train/therapeutic")])
	X_neg_train = np.array([np.load(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/train/marker/{filename}") \
				for filename in os.listdir(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/train/marker")])
	X_pos_test = np.array([np.load(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/test/therapeutic/{filename}") \
				for filename in os.listdir(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/test/therapeutic")])
	X_neg_test = np.array([np.load(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/test/marker/{filename}") \
				for filename in os.listdir(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/test/marker")])

	return X_pos_train, X_neg_train, X_pos_test, X_neg_test

def getDataSplit(X_pos, X_neg, numSamples=None):
	if numSamples is None:
		X = np.concatenate([
			np.take(X_pos, np.random.choice(len(X_pos), len(X_neg), replace=False), axis=0) if len(X_pos) > len(X_neg) else X_pos,
			np.take(X_neg, np.random.choice(len(X_neg), len(X_pos), replace=False), axis=0) if len(X_neg) > len(X_pos) else X_neg
		])
	else:
		X = np.concatenate([
			np.take(X_pos, np.random.choice(len(X_pos), numSamples, replace=False), axis=0),
			np.take(X_neg, np.random.choice(len(X_neg), numSamples, replace=False), axis=0)
		])

	Y = int(len(X) / 2) * [1] + int(len(X) / 2) * [0]

	return train_test_split(X, Y, test_size=0.1, stratify=Y)

def getUnbalancedDataSplit(X_pos, X_neg, numSamples=None):
	X = np.concatenate([X_pos, X_neg])

	Y = len(X_pos) * [1] + len(X_neg) * [0]

	return train_test_split(X, Y, test_size=0.1, stratify=Y)

def getDataSample(X_pos, X_neg, numSamples=None):
	if numSamples == None:
		X = np.concatenate([
			np.take(X_pos, np.random.choice(len(X_pos), len(X_neg), replace=False), axis=0) if len(X_pos) > len(X_neg) else X_pos,
			np.take(X_neg, np.random.choice(len(X_neg), len(X_pos), replace=False), axis=0) if len(X_neg) > len(X_pos) else X_neg
		])
	else:
		X = np.concatenate([
			np.take(X_pos, np.random.choice(len(X_pos), numSamples, replace=False), axis=0),
			np.take(X_neg, np.random.choice(len(X_neg), numSamples, replace=False), axis=0)
		])

	Y = np.array(int(len(X) / 2) * [1] + int(len(X) / 2) * [0])

	X, Y = unison_shuffled_copies(X, Y)

	return np.array(X), np.array(Y)

def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = list(np.random.permutation(len(a)))
	return [a[i] for i in p], [b[i] for i in p]

