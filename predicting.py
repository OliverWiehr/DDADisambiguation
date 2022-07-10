import os
import numpy as np

def predictDDAs(models, NDNs):
	markerIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/marker')]
	therapeuticIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/therapeutic')]
	
	predictedDDAs = []
	markerPredictions = 0
	therapeuticPredictions = 0
	
	if os.path.isfile('predictions.txt'):
		os.remove('predictions.txt')
	
	with open(f'files/predictedDDAs/allPredictedDDAs_2019_{NDNs}-NDNs-per-drug.txt', 'r') as ddaFile:
		numLines = sum(1 for line in ddaFile)

	lineNum = 0.0
	with open(f'files/predictedDDAs/allPredictedDDAs_2019_{NDNs}-NDNs-per-drug.txt', 'r') as ddaFile, open('predictions.txt', 'w') as predictionFile:
		for line in ddaFile:
			lineNum += 1
			print('Progress: %.2f%%' % (lineNum / numLines * 100), end='\r')
			
			info = line.split(";")
			ids = info[0].split("#")
			chemicalId = ids[0].split(":")[2]
			diseaseId = ids[1].split(":")[2]
			
			associationId = f"{chemicalId}_{diseaseId}"
			
			if info[1] != 'predicted' or associationId in markerIds or associationId in therapeuticIds:
				continue
			
			chemicalATCFiles = [file for file in os.listdir('vectors/ATCVectors') if file.startswith(chemicalId)]
			chemicalAHFSFiles = [file for file in os.listdir('vectors/AHFSVectors/chemical') if file.startswith(chemicalId)]
			diseaseAHFSFiles = [file for file in os.listdir('vectors/AHFSVectors/disease') if file.startswith(diseaseId)]
			
			if not os.path.isfile(f"vectors/vectors2019/chemical/{chemicalId}.npy") or not os.path.isfile(f"vectors/vectors2019/disease/{diseaseId}.npy"):
				continue
			
			chemicalVector = np.load(f"vectors/vectors2019/chemical/{chemicalId}.npy")
			diseaseVector = np.load(f"vectors/vectors2019/disease/{diseaseId}.npy")
			
			if associationId in predictedDDAs:
				continue
			
			predictedDDAs.append(associationId)
			
			associationVectors = []
			for chemicalAHFSFile in chemicalAHFSFiles:
				for chemicalATCFile in chemicalATCFiles:
					for diseaseAHFSFile in diseaseAHFSFiles:
						chemicalAHFSVector = np.load(f'vectors/AHFSVectors/chemical/{chemicalAHFSFile}')
						chemicalATCVector = np.load(f'vectors/ATCVectors/{chemicalATCFile}')
						diseaseAHFSVector = np.load(f'vectors/AHFSVectors/disease/{diseaseAHFSFile}')
						
						associationVectors.append(np.concatenate([chemicalVector, diseaseVector, chemicalAHFSVector, chemicalATCVector, diseaseAHFSVector]))
			
			if len(associationVectors) == 0:
				continue
			
			pubmedVectors = np.array([vector[:400] for vector in associationVectors])
			semanticVectors = np.array([vector[400:] for vector in associationVectors])

			if 51 in semanticVectors:
				continue
			
			pubmedVectors = pubmedVectors.reshape(pubmedVectors.shape[0], pubmedVectors.shape[1], 1)
			
			predictions = []
			for model in models:
				predictions += list(model.predict([pubmedVectors, semanticVectors]))
			
			averagePrediction = float(sum(1 for prediction in predictions if prediction[0] > 0.5)) / len(predictions)
			prediction = 'therapeutic' if averagePrediction > 0.5 else 'marker'
			
			ddaLine = line.strip("\n")
			predictionFile.write(f'{ddaLine};{prediction}\n')
			
			if prediction == 'therapeutic':
				therapeuticPredictions += 1
			else:
				markerPredictions += 1

	print(f'{therapeuticPredictions} therapeutic predictions made')
	print(f'{markerPredictions} marker predictions made')
