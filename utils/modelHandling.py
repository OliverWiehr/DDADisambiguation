import os
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.utils import plot_model
from .plotting import plotNetworkAccuracy, plotNetworkLoss, plotAverageConfusionMatrix, changeTheme

STANDARD_DATA_TYPE = 'concat'

def saveClassifier(type, classifier, prediction, labels, history, dataSource, dataType=STANDARD_DATA_TYPE, modelNum=0, id=0):
	with open(f'models/{type}_{dataSource}_{dataType}_{id}_{modelNum}.sav', 'wb') as classifierFile:
		pickle.dump(classifier, classifierFile)
	
	history['confusionMatrix'] = [
		  [sum(1 for i in range(len(prediction)) if prediction[i] >= 0.5 and labels[i] >= 0.5), # true positive
		   sum(1 for i in range(len(prediction)) if prediction[i] >= 0.5 and labels[i] < 0.5)], # false positive
		  [sum(1 for i in range(len(prediction)) if prediction[i] < 0.5 and labels[i] >= 0.5), # false negative
		   sum(1 for i in range(len(prediction)) if prediction[i] < 0.5 and labels[i] < 0.5)] # true negative
	]

	with open(f'history/{type}_{dataSource}_{dataType}_{id}_{modelNum}', 'wb') as historyFile:
		pickle.dump(history, historyFile)

	histories = loadClassifierHistories(type, dataSource, dataType, id)

	changeTheme('light')
	plt.figure()
	plotAverageConfusionMatrix(histories)
	plt.savefig(f'plots/confusion_{type}_{dataSource}_{dataType}_{id}.png')
	plt.close()
	changeTheme('dark')

def loadClassifier(type, dataSource, dataType=STANDARD_DATA_TYPE, modelNum=0, id=0):
	with open(f'models/{type}_{dataSource}_{dataType}_{id}_{modelNum}.sav', 'rb') as classifierFile:
		return pickle.load(classifierFile)

def saveNetwork(type, model, history, prediction, labels, dataSource, dataType=STANDARD_DATA_TYPE, modelNum=0, id=0):
	for path in ['plots']:
		if not os.path.exists(path):
			os.mkdir(path)
	
	plot_model(model, to_file=f'plots/model_{type}_{dataSource}_{dataType}_{id}.png', show_shapes=True, show_layer_names=False)
	
	history.history['confusionMatrix'] = [
		[sum(1 for i in range(len(prediction)) if prediction[i] >= 0.5 and labels[i] >= 0.5), # true positive
		 sum(1 for i in range(len(prediction)) if prediction[i] >= 0.5 and labels[i] < 0.5)], # false positive
		[sum(1 for i in range(len(prediction)) if prediction[i] < 0.5 and labels[i] >= 0.5), # false negative
		 sum(1 for i in range(len(prediction)) if prediction[i] < 0.5 and labels[i] < 0.5)] # true negative
	]

	with open(f'history/{type}_{dataSource}_{dataType}_{id}_{modelNum}', 'wb') as historyFile:
		pickle.dump(history.history, historyFile)
	
	changeTheme('light')
	histories = loadNetworkHistories(type, dataSource, dataType, id)
	plt.figure()
	plotNetworkAccuracy(histories)
	plt.savefig(f'plots/accuracy_{type}_{dataSource}_{dataType}_{id}.png')
	plt.close()
	plt.figure()
	plotNetworkLoss(histories)
	plt.savefig(f'plots/loss_{type}_{dataSource}_{dataType}_{id}.png')
	plt.close()
	plt.figure()
	plotAverageConfusionMatrix(histories)
	plt.savefig(f'plots/confusion_{type}_{dataSource}_{dataType}_{id}.png')
	plt.close()
	changeTheme('dark')


def loadPredictionNetworks(id=0):
	return [load_model(f'models/{file}') for file in os.listdir('models') if file.startswith(f'predictionModel_{id}')]

def loadNetwork(type, dataSource, dataType=STANDARD_DATA_TYPE, modelNum=0, id=0):
	return load_model(f'models/{type}_{dataSource}_{dataType}_{id}_{modelNum}.h5')

def loadClassifierHistories(type, dataSource, dataType=STANDARD_DATA_TYPE, id=0):
	histories = []
	for filename in [filename for filename in os.listdir('history') if filename.startswith(f'{type}_{dataSource}_{dataType}_{id}')]:
		with open(f'history/{filename}', 'rb') as historyFile:
			histories.append(pickle.load(historyFile))
	return histories

def loadNetworkHistories(type, dataSource, dataType=STANDARD_DATA_TYPE, id=0):
	histories = []
	for filename in [filename for filename in os.listdir('history') if filename.startswith(f'{type}_{dataSource}_{dataType}_{id}')]:
		with open(f'history/{filename}', 'rb') as historyFile:
			histories.append(pickle.load(historyFile))
	return histories
