import seaborn as sn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
import sys
from keras.backend import eval

theme = 'dark'
mpl.rcParams["savefig.dpi"] = 300

def changeTheme(newTheme):
	global theme
	
	theme = newTheme
	setTheme()

def setTheme():
	if theme == 'dark':
		textColor = 'white'
		backgroundColor = '#232323'
	else:
		textColor = 'black'
		backgroundColor = 'white'

	for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
		mpl.rcParams[param] = textColor
	mpl.rcParams['axes.facecolor'] = backgroundColor


def plotAverageClassifierAccuracy(histories):
	setTheme()
	accuracy = getAvgClassifierAccuracy(histories)
	plt.axis('off')
	plt.text(0,1,f"%.2f%%" % accuracy, fontsize=22)
	plt.text(0,0,f"Average accuracy")

def plotNetworkLoss(histories):
	setTheme()
#	plt.title("train vs validation loss")
	minEpochNum = min([len(history['acc']) for history in histories])
	for history in histories:
		plt.plot(history["loss"], color="blue", label="train", alpha=0.2)
		plt.plot(history["val_loss"], color="orange", label="validation", alpha=0.2)
		plt.plot(np.array([history['loss'][:minEpochNum] for history in histories]).mean(axis=0), color="blue", label="train")
		plt.plot(np.array([history['val_loss'][:minEpochNum] for history in histories]).mean(axis=0), color="orange", label="validation")
	plt.ylabel("loss")
	plt.xlabel("epoch")

def plotNetworkAccuracy(histories):
	setTheme()
#	plt.title("train vs validation accuracy")
	minEpochNum = min([len(history['acc']) for history in histories])
	for history in histories:
		plt.plot(history["acc"], color="blue", label="train", alpha=0.2)
		plt.plot(history["val_acc"], color="orange", label="validation", alpha=0.2)
		plt.plot(np.array([history['acc'][:minEpochNum] for history in histories]).mean(axis=0), color="blue", label="train")
		plt.plot(np.array([history['val_acc'][:minEpochNum] for history in histories]).mean(axis=0), color="orange", label="validation")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")

def plotAverageConfusionMatrix(histories):
	setTheme()
#	plt.title("avg confusion matrix")
	meanConfusionMatrix = [
		[int(np.mean([history['confusionMatrix'][0][0] for history in histories])),
		 int(np.mean([history['confusionMatrix'][0][1] for history in histories]))],
		[int(np.mean([history['confusionMatrix'][1][0] for history in histories])),
		 int(np.mean([history['confusionMatrix'][1][1] for history in histories]))]
	]
	df_cm = pd.DataFrame(meanConfusionMatrix, index = ['therapeutic', 'inducing'],
					  columns = ['therapeutic', 'inducing'])
	plt.ylabel("predictions")
	plt.xlabel("labels")
	sn.heatmap(df_cm, annot=True, fmt='d')

def plotNetworkMetrics(histories):
	setTheme()
	plt.axis('off')
	plt.text(0,1.0,f"%.2f%%" % (getAvgMaxNetworkAccuracy(histories)),
			fontsize=22)
	plt.text(0,0.95,f"avg max validation accuracy")

def plotClassifierMetrics(histories):
	setTheme()
	plt.axis('off')
	plt.text(0,1.0,f"%.2f%%" % (getAvgClassifierAccuracy(histories)),
			 fontsize=22)
	plt.text(0,0.95,f"avg validation accuracy")

def getAvgClassifierAccuracy(histories):
	return np.mean([history['accuracy'] for history in histories]) * 100

def getAvgMaxNetworkAccuracy(histories):
	return np.array([max(history['val_acc']) for history in histories]).mean(axis=0) * 100

def plotNetworkAttributes(model):
	setTheme()
	plt.axis('off')
	plt.text(0, .8, f'Learning rate: {eval(model.optimizer.lr)}', fontsize=12)

def analyzeNetwork(title, histories):
	setTheme()
	plt.figure(figsize=(18,0.2))
	plt.axis('off')
	plt.text(0,.7, title, fontsize=30)
	plt.show()

	fig = plt.figure(figsize=(16,5))
	plt.subplot(1,2,1)
	plotNetworkMetrics(histories)
	plt.subplot(1,2,2)
	plotAverageConfusionMatrix(histories)
	plt.show()

	fig = plt.figure(figsize=(16,5))
	plt.subplot(1,2,1)
	plotNetworkLoss(histories)
	plt.subplot(1,2,2)
	plotNetworkAccuracy(histories)
	plt.show()

def analyzeClassifier(title, histories):
	setTheme()
	plt.figure(figsize=(18,0.2))
	plt.axis('off')
	plt.text(0,.7, title, fontsize=30)
	plt.show()
	
	fig = plt.figure(figsize=(16,5))
	plt.subplot(1,2,1)
	plotClassifierMetrics(histories)
	plt.subplot(1,2,2)
	plotAverageConfusionMatrix(histories)
	plt.show()

def analyzePredictionModels(title, histories):
	setTheme()
	plt.figure(figsize=(18,0.2))
	plt.axis('off')
	plt.text(0,.7, title, fontsize=30)
	plt.show()
	
	fig = plt.figure(figsize=(16,5))
	plt.subplot(1,2,1)
	plt.text(0,0.2,f"%.2f%%" % (getAvgMaxNetworkAccuracy(histories)),
			 fontsize=22)
	plt.text(0,0,f"avg max validation accuracy")
	plt.show()
	fig = plt.figure(figsize=(16,5))
	plt.subplot(1,2,1)
	plt.title("train vs validation loss")
	for history in histories:
		plt.plot(history["loss"], color="blue", label="train")
		plt.plot(history["val_loss"], color="orange", label="validation")
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.subplot(1,2,2)
	plt.title("train vs validation accuracy")
	for history in histories:
		plt.plot(history["acc"], color="blue", label="train")
		plt.plot(history["val_acc"], color="orange", label="validation")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.show()

def compareModels(title, metric, values, accuracies):
	setTheme()
	plt.figure(figsize=(18,1.5))
	plt.axis('off')
	plt.text(0,1, title, fontsize=26)

	plt.text(0,0.4, metric, fontsize=18)
	for i in range(len(values)):
		plt.text(0.1 * (i+1),0.4, values[i], fontsize=18)

	plt.text(0,0,"accuracy", fontsize=18)
	for i in range(len(values)):
		plt.text(0.1 * (i+1),0, "%.2f" % (accuracies[i]), fontsize=18)

	plt.show()

def compareNetworks(titel1, titel2, histories1, histories2, model1=None, model2=None):
	setTheme()
	fig = plt.figure(figsize=(16,.5))
	plt.axis('off')
	plt.text(0, 0, titel1, fontsize=30)
	plt.text(.5, 0, titel2, fontsize=30)
	plt.show()


	if model1 is not None and model2 is not None:
		old_stdout = sys.stdout
		model1Output = StringIO()
		sys.stdout = model1Output
		print(model1.summary())
		model2Output = StringIO()
		sys.stdout = model2Output
		print(model2.summary())
		sys.stdout = old_stdout
		model1Summary = model1Output.getvalue().replace('=', '')
		model2Summary = model2Output.getvalue().replace('=', '')

		fig = plt.figure(figsize=(16,3))
		plt.axis('off')
		plt.text(0, 0, model1Summary, fontsize=12)
		plt.text(.5, 0, model2Summary, fontsize=12)
		plt.show()

		fig = plt.figure(figsize=(16, .5))
		plt.subplot(1,2,1)
		plotNetworkAttributes(model1)
		plt.subplot(1,2,2)
		plotNetworkAttributes(model2)
		plt.show()

	fig = plt.figure(figsize=(16, 1))
	plt.subplot(1,2,1)
	plotNetworkMetrics(histories1)
	plt.subplot(1,2,2)
	plotNetworkMetrics(histories2)
	plt.show()

	fig = plt.figure(figsize=(16,5))
	plt.subplot(1,2,1)
	plotNetworkLoss(histories1)
	plt.subplot(1,2,2)
	plotNetworkLoss(histories2)
	plt.show()

	fig = plt.figure(figsize=(16,5))
	plt.subplot(1,2,1)
	plotNetworkAccuracy(histories1)
	plt.subplot(1,2,2)
	plotNetworkAccuracy(histories2)
	plt.show()

	fig = plt.figure(figsize=(16,5))
	plt.subplot(1,2,1)
	plotAverageConfusionMatrix(histories1)
	plt.subplot(1,2,2)
	plotAverageConfusionMatrix(histories2)
	plt.show()
