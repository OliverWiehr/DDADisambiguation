from .modelHandling import saveClassifier, saveNetwork, loadNetwork
from .dataLoading import getDataSplit, getDataSample, loadRetrospectiveAnalysisDataWithSemantics, loadPredictedDDAsWithSemantics

from sklearn import svm
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Conv2D, Dropout, MaxPooling1D, MaxPooling2D, Flatten, Input, concatenate, Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

STANDARD_DATA_TYPE = 'concat'

def getCallbacks(type, dataSource, dataType, id, modelNum):
	earlyStoppingCallback = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=20)
	modelCheckpointCallback = ModelCheckpoint(f'models/{type}_{dataSource}_{dataType}_{id}_{modelNum}.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
	return [earlyStoppingCallback, modelCheckpointCallback]


def trainLCF(X_pos, X_neg, dataSource, dataType=STANDARD_DATA_TYPE, X_pos_test=None, X_neg_test=None, degree=3, numModels=10, id=0, numSamples=None):
	for modelNum in range(numModels):
		print(f"Training classifier {modelNum + 1} / {numModels}")

		classifier = svm.SVC(degree=degree)

		if X_pos_test is None or X_neg_test is None:
			X_train, X_test, y_train, y_test = getDataSplit(X_pos, X_neg, numSamples)
		else:
			X_train, y_train = getDataSample(X_pos, X_neg, numSamples)
			X_test, y_test = getDataSample(X_pos_test, X_neg_test, numSamples)

		classifier.fit(X_train, y_train)
		accuracy = classifier.score(X_test, y_test)

		prediction = classifier.predict(X_test)
		print(f"Accuracy: {accuracy}")

		saveClassifier('lcf', classifier, prediction, y_test, {'accuracy': accuracy}, dataSource, dataType, modelNum, id)

def trainANN(X_pos, X_neg, dataSource, dataType=STANDARD_DATA_TYPE, X_pos_test=None, X_neg_test=None, batchSize=8, epochs=1000, lr=1e-4, numModels=10, id=0, numSamples=None):
	optimizer = Adam(lr=lr)

	modelNum = 0
	while modelNum < numModels:
		print(f"Training model {modelNum+1} / {numModels}")

		model = Sequential()
		
		model.add(Dense(400, activation='relu', input_dim=X_pos.shape[1]))
		model.add(Dense(200, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))

		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

		if X_pos_test is None or X_neg_test is None:
			X_train, X_test, y_train, y_test = getDataSplit(X_pos, X_neg, numSamples=numSamples)
		else:
			X_train, y_train = getDataSample(X_pos, X_neg, numSamples=numSamples)
			X_test, y_test = getDataSample(X_pos_test, X_neg_test, numSamples=numSamples)
		
		callbacks = getCallbacks('ann', dataSource, dataType, id, modelNum)
		history = model.fit(X_train, y_train, batch_size=batchSize, validation_data=(X_test, y_test), epochs=epochs, callbacks=callbacks)
		if min(history.history['val_acc']) < 0.51:
			continue
		
		model = loadNetwork('ann', dataSource, dataType, modelNum, id)
		prediction = model.predict(X_test)

		saveNetwork('ann', model, history, prediction, y_test, dataSource, dataType, modelNum, id)
		modelNum += 1

def trainCNN(X_pos, X_neg, dataSource, dataType=STANDARD_DATA_TYPE, X_pos_test=None, X_neg_test=None, batchSize=8, epochs=1000, lr=1e-4, numModels=10, id=0, numSamples=None):
	optimizer = Adam(lr=lr)

	modelNum = 0
	while modelNum < numModels:
		print(f"Training model {modelNum+1} / {numModels}")

		if X_pos_test is None or X_neg_test is None:
			X_train, X_test, y_train, y_test = getDataSplit(X_pos, X_neg, numSamples=numSamples)
		else:
			X_train, y_train = getDataSample(X_pos, X_neg, numSamples=numSamples)
			X_test, y_test = getDataSample(X_pos_test, X_neg_test, numSamples=numSamples)
		
		model = Sequential()
		if len(X_train.shape) == 3:
			X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
			X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
			
			model.add(Conv2D(filters=64, kernel_size=(2, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
			model.add(Conv2D(filters=64, kernel_size=(1, 3), activation='relu'))
		else:
			X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
			X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
			
			model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
			model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
		model.add(Dropout(0.1))
		if len(X_train.shape) == 4:
			model.add(MaxPooling2D(pool_size=(1, 2)))
		else:
			model.add(MaxPooling1D(pool_size=2))
		model.add(Flatten())
		model.add(Dense(100, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

		callbacks = getCallbacks('cnn', dataSource, dataType, id, modelNum)
		history = model.fit(X_train, y_train, batch_size=batchSize, validation_data=(X_test, y_test), epochs=epochs, callbacks = callbacks)
		if min(history.history['val_acc']) < 0.51:
			continue
		
		model = loadNetwork('cnn', dataSource, dataType, modelNum, id)
		prediction = model.predict(X_test)

		saveNetwork('cnn', model, history, prediction, y_test, dataSource, dataType, modelNum, id)
		modelNum += 1


def trainHybridANN(dataSource, NDNs=None, dataType=STANDARD_DATA_TYPE, batchSize=8, epochs=1000, lr=1e-4, numModels=10, id=0, numSamples=None):
	optimizer = Adam(lr=lr)
	
	if dataSource.startswith('retrospectiveAnalysisWithSemantics'):
		X_pos_train, X_neg_train, X_pos_test, X_neg_test = loadRetrospectiveAnalysisDataWithSemantics(NDNs, dataType)
	elif dataSource.startswith('predictedDDAs'):
		X_pos_train, X_neg_train, X_pos_test, X_neg_test = loadPredictedDDAsWithSemantics(NDNs, dataType)

	modelNum = 0
	while modelNum < numModels:
		print(f"Training model {modelNum+1} / {numModels}")
		
		if dataSource == 'semanticCTDDDAsWithPubmed':
			X_pos_train, X_neg_train, X_pos_test, X_neg_test = loadSemanticCTDDDAsWithPubmed()
		
		X_train, y_train = getDataSample(X_pos_train, X_neg_train)
		X_test, y_test = getDataSample(X_pos_test, X_neg_test)

		vectors_train = np.array([dataPoint[:400] for dataPoint in X_train])
		semantics_train = np.array([dataPoint[400:] for dataPoint in X_train])
		
		vectors_test = np.array([dataPoint[:400] for dataPoint in X_test])
		semantics_test = np.array([dataPoint[400:] for dataPoint in X_test])
		
		vectorInputs = Input(shape=(vectors_train.shape[1],))
		vectorModel = Dense(400, activation='relu', input_dim=400)(vectorInputs)
		vectorModel = Dense(200, activation='relu')(vectorModel)
		vectorModel = Model(vectorInputs, vectorModel)
		
		semanticInputs = Input(shape=(semantics_train.shape[1],))
		semanticModel = Embedding(1064, 128, input_length=28)(semanticInputs)
		semanticModel = Flatten()(semanticModel)
		semanticModel = Dense(28, activation='relu', input_dim=semantics_train.shape[1])(semanticModel)
		semanticModel = Dense(14, activation='relu')(semanticModel)
		semanticModel = Model(semanticInputs, semanticModel)
		
		combinedOutput = concatenate([vectorModel.output, semanticModel.output])
		
		mergingModel = Dense(214, activation='relu')(combinedOutput)
		mergingModel = Dense(107, activation='relu')(mergingModel)
		mergingModel = Dense(1, activation='sigmoid')(mergingModel)
		
		model = Model([vectorModel.input, semanticModel.input], mergingModel)
		
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

		callbacks = getCallbacks('hybridANN', dataSource, dataType, id, modelNum)
		history = model.fit([vectors_train, semantics_train], y_train, batch_size=batchSize, validation_data=([vectors_test, semantics_test], y_test), epochs=epochs, callbacks = callbacks)
		if min(history.history['val_acc']) < 0.51:
			continue
		
		model = loadNetwork('hybridANN', dataSource, dataType, modelNum, id)
		prediction = model.predict([vectors_test, semantics_test])

		saveNetwork('hybridANN', model, history, prediction, y_test, dataSource, dataType, modelNum, id)
		modelNum += 1

def trainHybridCNN(dataSource, NDNs=None, dataType=STANDARD_DATA_TYPE, batchSize=8, epochs=1000, lr=1e-4, numModels=10, id=0):
	optimizer = Adam(lr=lr)
	
	if dataSource.startswith('retrospectiveAnalysisWithSemantics'):
		X_pos_train, X_neg_train, X_pos_test, X_neg_test = loadRetrospectiveAnalysisDataWithSemantics(NDNs, dataType)
	elif dataSource.startswith('predictedDDAs'):
		X_pos_train, X_neg_train, X_pos_test, X_neg_test = loadPredictedDDAsWithSemantics(NDNs, dataType)

	modelNum = 0
	while modelNum < numModels:
		print(f"Training model {modelNum+1} / {numModels}")

		if dataSource == 'semanticCTDDDAsWithPubmed':
			X_pos_train, X_neg_train, X_pos_test, X_neg_test = loadSemanticCTDDDAsWithPubmed()
		
		X_train, y_train = getDataSample(X_pos_train, X_neg_train)
		X_test, y_test = getDataSample(X_pos_test, X_neg_test)

		vectors_train = np.array([dataPoint[:400] for dataPoint in X_train])
		semantics_train = np.array([dataPoint[400:] for dataPoint in X_train])

		vectors_test = np.array([dataPoint[:400] for dataPoint in X_test])
		semantics_test = np.array([dataPoint[400:] for dataPoint in X_test])

		vectors_test = vectors_test.reshape(vectors_test.shape[0], vectors_test.shape[1], 1)
		vectors_train = vectors_train.reshape(vectors_train.shape[0], vectors_train.shape[1], 1)

		vectorInputs = Input(shape=(vectors_train.shape[1], vectors_train.shape[2]))
		vectorModel = Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(vectors_train.shape[1], vectors_train.shape[2]))(vectorInputs)
		vectorModel = Conv1D(filters=64, kernel_size=3, activation='relu')(vectorModel)
		vectorModel = Dropout(0.1)(vectorModel)
		vectorModel = MaxPooling1D(pool_size=2)(vectorModel)
		vectorModel = Flatten()(vectorModel)
		vectorModel = Dense(100, activation='relu')(vectorModel)
		vectorModel = Model(vectorInputs, vectorModel)

		semanticInputs = Input(shape=(semantics_train.shape[1],))
		semanticModel = Embedding(1064, 128, input_length=28)(semanticInputs)
		semanticModel = Conv1D(64, 5, activation='relu', input_shape=(semantics_train.shape[1], 128))(semanticModel)
		semanticModel = MaxPooling1D(5)(semanticModel)
		semanticModel = Flatten()(semanticModel)
		semanticModel = Dense(14, activation='relu')(semanticModel)
		semanticModel = Model(semanticInputs, semanticModel)

		combinedOutput = concatenate([vectorModel.output, semanticModel.output])

		mergingModel = Dense(214, activation='relu')(combinedOutput)
		mergingModel = Dense(107, activation='relu')(mergingModel)
		mergingModel = Dense(1, activation='sigmoid')(mergingModel)

		model = Model([vectorModel.input, semanticModel.input], mergingModel)

		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

		callbacks = getCallbacks('hybridCNN', dataSource, dataType, id, modelNum)
		history = model.fit([vectors_train, semantics_train], y_train, batch_size=batchSize, validation_data=([vectors_test, semantics_test], y_test), epochs=epochs, callbacks=callbacks)
		if min(history.history['val_acc']) < 0.51:
			continue
		
		model = loadNetwork('hybridCNN', dataSource, dataType, modelNum, id)
		prediction = model.predict([vectors_test, semantics_test])

		saveNetwork('hybridCNN', model, history, prediction, y_test, dataSource, dataType, modelNum, id)
		modelNum += 1

def trainHybrid(dataSource, NDNs=None, dataType=STANDARD_DATA_TYPE, batchSize=8, epochs=1000, lr=1e-4, numModels=10, id=0, numSamples=None):
	optimizer = Adam(lr=lr)
	
	if dataSource.startswith('retrospectiveAnalysisWithSemantics'):
		X_pos_train, X_neg_train, X_pos_test, X_neg_test = loadRetrospectiveAnalysisDataWithSemantics(NDNs, dataType)
	elif dataSource.startswith('predictedDDAs'):
		X_pos_train, X_neg_train, X_pos_test, X_neg_test = loadPredictedDDAsWithSemantics(NDNs, dataType)
	
	modelNum = 0
	while modelNum < numModels:
		print(f"Training model {modelNum+1} / {numModels}")
		
		if dataSource == 'semanticCTDDDAsWithPubmed':
			X_pos_train, X_neg_train, X_pos_test, X_neg_test = loadSemanticCTDDDAsWithPubmed()
		
		X_train, y_train = getDataSample(X_pos_train, X_neg_train)
		X_test, y_test = getDataSample(X_pos_test, X_neg_test)
		
		vectors_train = np.array([dataPoint[:400] for dataPoint in X_train])
		semantics_train = np.array([dataPoint[400:] for dataPoint in X_train])
		
		vectors_test = np.array([dataPoint[:400] for dataPoint in X_test])
		semantics_test = np.array([dataPoint[400:] for dataPoint in X_test])
		
		vectorInputs = Input(shape=(vectors_train.shape[1],))
		vectorModel = Dense(400, activation='relu', input_dim=X_pos_train.shape[1])(vectorInputs)
		vectorModel = Dense(200, activation='relu')(vectorModel)
		vectorModel = Model(vectorInputs, vectorModel)
		
		semanticInputs = Input(shape=(semantics_train.shape[1],))
		semanticModel = Embedding(1064, 128, input_length=28)(semanticInputs)
		semanticModel = Conv1D(64, 5, activation='relu', input_shape=(semantics_train.shape[1], 128))(semanticModel)
		semanticModel = MaxPooling1D(5)(semanticModel)
		semanticModel = Flatten()(semanticModel)
		semanticModel = Dense(14, activation='relu')(semanticModel)
		semanticModel = Model(semanticInputs, semanticModel)
		
		combinedOutput = concatenate([vectorModel.output, semanticModel.output])
		
		mergingModel = Dense(214, activation='relu')(combinedOutput)
		mergingModel = Dense(107, activation='relu')(mergingModel)
		mergingModel = Dense(1, activation='sigmoid')(mergingModel)
		
		model = Model([vectorModel.input, semanticModel.input], mergingModel)
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
		
		callbacks = getCallbacks('hybrid', dataSource, dataType, id, modelNum)
		history = model.fit([vectors_train, semantics_train], y_train, batch_size=batchSize, validation_data=([vectors_test, semantics_test], y_test), epochs=epochs, callbacks=callbacks)
		if min(history.history['val_acc']) < 0.51:
			continue
		
		model = loadNetwork('hybrid', dataSource, dataType, modelNum, id)
		prediction = model.predict([vectors_test, semantics_test])
		
		saveNetwork('hybrid', model, history, prediction, y_test, dataSource, dataType, modelNum, id)
		modelNum += 1


def trainPredictionModel(batchSize=256, epochs=1000, lr=1e-5, numModels=10, id=0):
	optimizer = Adam(lr=lr)
	
	modelNum = 0
	while modelNum < numModels:
		print(f"Training model {modelNum+1} / {numModels}")
		
		X_pos_train, X_neg_train, X_pos_test, X_neg_test = loadSemanticCTDDDAsWithPubmed()
	
		X_train, y_train = getDataSample(X_pos_train, X_neg_train)
		X_test, y_test = getDataSample(X_pos_test, X_neg_test)
		
		vectors_train = np.array([dataPoint[:400] for dataPoint in X_train])
		semantics_train = np.array([dataPoint[400:] for dataPoint in X_train])
		
		vectors_test = np.array([dataPoint[:400] for dataPoint in X_test])
		semantics_test = np.array([dataPoint[400:] for dataPoint in X_test])
		
		vectors_test = vectors_test.reshape(vectors_test.shape[0], vectors_test.shape[1], 1)
		vectors_train = vectors_train.reshape(vectors_train.shape[0], vectors_train.shape[1], 1)
		
		vectorInputs = Input(shape=(vectors_train.shape[1], vectors_train.shape[2]))
		vectorModel = Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(vectors_train.shape[1], vectors_train.shape[2]))(vectorInputs)
		vectorModel = Conv1D(filters=64, kernel_size=3, activation='relu')(vectorModel)
		vectorModel = Dropout(0.1)(vectorModel)
		vectorModel = MaxPooling1D(pool_size=2)(vectorModel)
		vectorModel = Flatten()(vectorModel)
		vectorModel = Dense(100, activation='relu')(vectorModel)
		vectorModel = Model(vectorInputs, vectorModel)
		
		semanticInputs = Input(shape=(semantics_train.shape[1],))
		semanticModel = Embedding(1064, 128, input_length=28)(semanticInputs)
		semanticModel = Conv1D(128, 5, activation='relu', input_shape=(semantics_train.shape[1], 128))(semanticModel)
		semanticModel = MaxPooling1D(5)(semanticModel)
		semanticModel = Flatten()(semanticModel)
		semanticModel = Dense(128, activation='relu')(semanticModel)
		semanticModel = Model(semanticInputs, semanticModel)
		
		combinedOutput = concatenate([vectorModel.output, semanticModel.output])
		
		mergingModel = Dense(228, activation='relu')(combinedOutput)
		mergingModel = Dense(114, activation='relu')(mergingModel)
		mergingModel = Dense(1, activation='sigmoid')(mergingModel)
		
		model = Model([vectorModel.input, semanticModel.input], mergingModel)
		
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
		
		history = model.fit([vectors_train, semantics_train], y_train, batch_size=batchSize, validation_data=([vectors_test, semantics_test], y_test), epochs=epochs, callbacks=callbacks)
		if min(history.history['val_acc']) < 0.51:
			continue
		
		model = loadNetwork('preditctionModel', dataSource, dataType, modelNum, id)
		prediction = model.predict([vectors_test, semantics_test])
		
		saveNetwork('predictionModel', model, history, prediction, y_test, 'CTDDDAs', 'regular', modelNum, id)
		modelNum += 1
