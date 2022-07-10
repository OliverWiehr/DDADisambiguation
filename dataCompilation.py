import os
import shutil
import numpy as np
import javaobj
from gensim.models import KeyedVectors
import re
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

STANDARD_DATA_TYPE = 'concat'

def generateAssociationVector(chemicalVector, diseaseVector, dataType):
	if dataType == 'mean':
		return (chemicalVector + diseaseVector) / 2
	if dataType == 'concat':
		return np.concatenate([chemicalVector, diseaseVector])
	if dataType == 'append':
		return np.array([chemicalVector, diseaseVector])

def compileVectors():
	files = {
		1989: 'files/pubpharmVector_200dims_20window_1iter_5min_stop_w_filtered_100percent_PubMed_sample_1989-01-01.txt', \
		2019: 'files/drugDiseaseVectors100Pubmed2019.txt' \
	}

	if not os.path.exists('vectors'):
		os.mkdir('vectors')

	for year in [1989, 2019]:
		if os.path.exists(f"vectors/ectors{year}"):
			shutil.rmtree(f"vectors/vectors{year}")

		for directory in [f"vectors/vectors{year}", f"vectors/vectors{year}/chemical", f"vectors/vectors{year}/disease"]:
			os.mkdir(directory)

		numChemicalVectors = 0
		numDiseaseVectors = 0

		with open(files[year], 'r') as vectorFile:
			for line in vectorFile:
				elements = line.strip("\n").split(" ")
				info = elements[0]

				try:
					label = info.split(':')[0]
				except IndexError:
					continue

				for id in re.compile('[a-z]\d{6}').findall(info):
					if label == 'chemical':
						numChemicalVectors += 1
					elif label == 'disease':
						numDiseaseVectors += 1
					else: 
						continue

					vector = np.array([float(el) for el in elements[1:]])

					np.save(f"vectors/vectors{year}/{label}/{id}.npy", vector)

		print(year)
		print(f"\t{numDiseaseVectors} disease vectors")
		print(f"\t{numChemicalVectors} chemical vectors")

def compileVectorsWithReducedDim():
	if not os.path.exists('vectors'):
		os.mkdir('vectors')

	for year in [1989, 2019]:
		print(year)

		vectors = []
		paths = []
			
		for category in ['chemical', 'disease']:
			for filename in os.listdir(f'vectors/vectors{year}/{category}'):
				vectors.append(np.load(f'vectors/vectors{year}/{category}/{filename}'))
				paths.append(f'{category}/{filename}')

		pca = PCA()
		pca.fit(vectors)
		cumsum = np.cumsum(pca.explained_variance_ratio_)
		d = np.argmax(cumsum >= 0.95) + 1

		pca = PCA(n_components=d)
		pcvectors = pca.fit_transform(vectors)

		if os.path.exists(f"vectors/vectors{year}-reducedDim"):
			shutil.rmtree(f"vectors/vectors{year}-reducedDim")

		for directory in [f"vectors/vectors{year}-reducedDim", f"vectors/vectors{year}-reducedDim/chemical", f"vectors/vectors{year}-reducedDim/disease"]:
			os.mkdir(directory)

		for i in range(len(pcvectors)):
			np.save(f"vectors/vectors{year}-reducedDim/{paths[i]}", pcvectors[i])

		print(f'\tOriginal dimension: 200')
		print(f'\Reduced dimensions: {d}\n')

def compileSparseMeSHVectors():
	if not os.path.exists('vectors'):
		os.mkdir('vectors')

	numChemicalVectors = 0
	numDiseaseVectors = 0

	drugRootValues = set()
	with open('files/drugMeshTreesATCandMeSH.txt', 'r') as meshTreeFile:
		for line in meshTreeFile:
			elements = line.split('#')
			info = elements[0].split(':')
			if len(info) != 2 or info[0] != 'chemical':
				continue
			
			trees = elements[1].strip('[').strip(']').split(', ')
			roots = {tree.split('.')[0].split('/')[0] for tree in trees if len(tree.split('.')[0].split('/')[0]) > 0}

			drugRootValues.update(roots)

	drugRootValues = list(drugRootValues)
	drugRootValues = dict([(drugRootValues[i], i) for i in range(len(drugRootValues))])

	if os.path.exists('vectors/sparseMeSHVectors'):
		shutil.rmtree('vectors/sparseMeSHVectors')

	for directory in ['vectors/sparseMeSHVectors', 'vectors/sparseMeSHVectors/chemical', 'vectors/sparseMeSHVectors/disease']:
		os.mkdir(directory)

	with open('files/drugMeshTreesATCandMeSH.txt', 'r') as meshTreeFile:
		for line in meshTreeFile:
			rootVector = np.array([0] * len(drugRootValues))

			elements = line.split('#')
			info = elements[0].split(':')
			if len(info) != 2 or info[0] != 'chemical':
				continue
			
			ids = info[1].split('|')

			trees = elements[1].strip('[').strip(']').split(', ')
			roots = {tree.split('.')[0].split('/')[0] for tree in trees if len(tree.split('.')[0].split('/')[0]) > 0}

			for root in roots:
				rootVector[drugRootValues[root]] = 1

			for id in ids:
				np.save(f'vectors/sparseMeSHVectors/chemical/{id}.npy', rootVector)
				numChemicalVectors += 1

	diseaseRootValues = set()
	with open('files/diseaseMeshTrees.txt', 'r') as meshTreeFile:
		for line in meshTreeFile:
			elements = line.split('#')
			info = elements[0].split(':')
			if len(info) != 2 or info[0] != 'disease':
				continue
			
			trees = elements[1].strip('[').strip(']').split(', ')
			roots = {tree.split('.')[0].split('/')[0].strip(']\n') for tree in trees if len(tree.split('.')[0].split('/')[0]) > 0}

			diseaseRootValues.update(roots)

	diseaseRootValues = list(diseaseRootValues)
	diseaseRootValues = dict([(diseaseRootValues[i], i) for i in range(len(diseaseRootValues))])

	with open('files/diseaseMeshTrees.txt', 'r') as meshTreeFile:
		for line in meshTreeFile:
			rootVector = [0] * len(diseaseRootValues)

			elements = line.split('#')
			info = elements[0].split(':')
			if len(info) != 2 or info[0] != 'disease':
				continue
			
			id = info[1]
			
			trees = elements[1].strip('[').strip(']').split(', ')
			roots = {tree.split('.')[0].split('/')[0].strip(']\n') for tree in trees if len(tree.split('.')[0].split('/')[0]) > 0}

			for root in roots:
				rootVector[diseaseRootValues[root]] = 1

			np.save(f'vectors/sparseMeSHVectors/disease/{id}.npy', rootVector)
			numDiseaseVectors += 1

	print(f'{numChemicalVectors} chemical vectors')
	print(f'{numDiseaseVectors} disease vectors')

def compileDenseMeSHVectors():
	if not os.path.exists('vectors'):
		os.mkdir('vectors')

	if os.path.exists("vectors/denseMeSHVectors"):
		shutil.rmtree("vectors/denseMeSHVectors")

	for directory in ["vectors/denseMeSHVectors", "vectors/denseMeSHVectors/chemical", "vectors/denseMeSHVectors/disease"]:
		os.mkdir(directory)

	for category in ['chemical', 'disease']:
		print(category)

		vectors = []
		ids = []

		originalDimsSet = False
		for file in os.listdir(f'vectors/sparseMeSHVectors/{category}'):
			vectors.append(np.load(f'vectors/sparseMeSHVectors/{category}/{file}'))
			ids.append(file)

			if not originalDimsSet:
				originalDimensions = np.load(f'vectors/sparseMeSHVectors/{category}/{file}').shape[0]
				originalDimsSet = True

		pca = PCA()
		pca.fit(vectors)
		cumsum = np.cumsum(pca.explained_variance_ratio_)
		d = np.argmax(cumsum >= 0.95) + 1

		pca = PCA(n_components=d)
		sc_vectors = pca.fit_transform(vectors)

		vectors = pca.transform(vectors)

		numVectors = 0

		for i in range(len(vectors)):
			vector = vectors[i]
			id = ids[i]

			np.save(f"vectors/denseMeSHVectors/{category}/{id}", vector)

			numVectors += 1

		print(f"\t{originalDimensions} original dimensions")
		print(f"\t{d} compressed dimensions\n")
		print(f"\t{numVectors} vectors\n")

def compileSVDMeSHVectors():
	if not os.path.exists('vectors'):
		os.mkdir('vectors')

	if os.path.exists("vectors/SVDMeSHVectors"):
		shutil.rmtree("vectors/SVDMeSHVectors")

	for directory in ["vectors/SVDMeSHVectors", "vectors/SVDMeSHVectors/chemical", "vectors/SVDMeSHVectors/disease"]:
		os.mkdir(directory)

	for category in ['chemical', 'disease']:
		print(category)

		vectors = []
		ids = []

		for file in os.listdir(f'vectors/sparseMeSHVectors/{category}'):
			vectors.append(np.load(f'vectors/sparseMeSHVectors/{category}/{file}'))
			ids.append(file)

		vectors = np.array(vectors)

		originalDimensions = vectors.shape[1]

		svd = TruncatedSVD(n_components=5)
		svd.fit(vectors)
		vectors = svd.transform(vectors)

		numVectors = 0

		for i in range(len(vectors)):
			vector = vectors[i]
			id = ids[i]

			np.save(f"vectors/SVDMeSHVectors/{category}/{id}", vector)

			numVectors += 1

		compressedDimensions = vectors.shape[1]

		print(f"\t{originalDimensions} original dimensions")
		print(f"\t{compressedDimensions} compressed dimensions\n")
		print(f"\t{numVectors} vectors\n")

def compileSparseATCVectors():
	if not os.path.exists('vectors'):
		os.mkdir('vectors')

	numVectors = 0

	drugRsaValues = set()
	with open('files/drugMeshTreesATCandMeSH.txt', 'r') as meshTreeFile:
		for line in meshTreeFile:
			elements = line.split('#')
			info = elements[0].split(':')
			if len(info) != 2 or info[0] != 'chemical':
				continue
			
			rsaSequences = elements[2].strip('[').strip(']').split(', ')
			rsas = {rsa[0] for rsa in rsaSequences if len(rsa) > 0}

			drugRsaValues.update(rsas)

	drugRsaValues = list(drugRsaValues)
	drugRsaValues = dict([(drugRsaValues[i], i) for i in range(len(drugRsaValues))])

	if os.path.exists('vectors/sparseATCVectors'):
		shutil.rmtree('vectors/sparseATCVectors')

	os.mkdir('vectors/sparseATCVectors')

	with open('files/drugMeshTreesATCandMeSH.txt', 'r') as meshTreeFile:
		for line in meshTreeFile:
			rsaVector = np.array([0] * len(drugRsaValues))

			elements = line.split('#')
			info = elements[0].split(':')
			if len(info) != 2 or info[0] != 'chemical':
				continue
			
			ids = info[1].split('|')
			
			rsaSequences = elements[2].strip('[').strip(']').split(', ')
			rsas = {rsa[0] for rsa in rsaSequences if len(rsa) > 0}

			for rsa in rsas:
				rsaVector[drugRsaValues[rsa]] = 1

			for id in ids:
				np.save(f'vectors/sparseATCVectors/{id}.npy', rsaVector)
				numVectors += 1

	print(f'{numVectors} vectors')

def compileSVDATCVectors():
	if not os.path.exists('vectors'):
		os.mkdir('vectors')

	vectors = []
	ids = []

	originalDimsSet = False
	for file in os.listdir(f'vectors/sparseATCVectors'):
		vectors.append(np.load(f'vectors/sparseATCVectors/{file}'))
		ids.append(file)

		if not originalDimsSet:
			originalDimensions = np.load(f'vectors/sparseATCVectors/{file}').shape[0]
			originalDimsSet = True

	vectors = np.array(vectors)

	originalDimensions = vectors.shape[1]

	svd = TruncatedSVD(n_components=5)
	svd.fit(vectors)
	vectors = svd.transform(vectors)

	compressedDimensions = vectors.shape[1]

	if os.path.exists("vectors/denseATCVectors"):
		shutil.rmtree("vectors/denseATCVectors")

	os.mkdir('vectors/denseATCVectors')

	numVectors = 0

	for i in range(len(vectors)):
		vector = vectors[i]
		id = ids[i]

		np.save(f"vectors/denseATCVectors/{id}", vector)

		numVectors += 1

	print(f"{originalDimensions} original dimensions")
	print(f"{compressedDimensions} compressed dimensions\n")
	print(f"{numVectors} vectors")

def compileDenseATCVectors():
	if not os.path.exists('vectors'):
		os.mkdir('vectors')

	vectors = []
	ids = []

	originalDimsSet = False
	for file in os.listdir(f'svectors/parseATCVectors'):
		vectors.append(np.load(f'vectors/sparseATCVectors/{file}'))
		ids.append(file)

		if not originalDimsSet:
			originalDimensions = np.load(f'vectors/sparseATCVectors/{file}').shape[0]
			originalDimsSet = True

	pca = PCA()
	pca.fit(vectors)
	cumsum = np.cumsum(pca.explained_variance_ratio_)
	d = np.argmax(cumsum >= 0.95) + 1

	pca = PCA(n_components=d)
	sc_vectors = pca.fit_transform(vectors)

	vectors = pca.transform(vectors)

	if os.path.exists("vectors/SVDATCVectors"):
		shutil.rmtree("vectors/SVDATCVectors")

	os.mkdir('vectors/SVDATCVectors')

	numVectors = 0

	for i in range(len(vectors)):
		vector = vectors[i]
		id = ids[i]

		np.save(f"vectors/SVDATCVectors/{id}", vector)

		numVectors += 1

	print(f"{originalDimensions} original dimensions")
	print(f"{d} compressed dimensions\n")
	print(f"{numVectors} vectors")

def compileCTDDDAs(dataTypes=[STANDARD_DATA_TYPE], reducedDim=False):
	if not os.path.exists("data"):
		os.mkdir("data")
	
	for dataType in dataTypes:
		if os.path.exists(f"data/CTDDDAs_{dataType}{'-reducedDim' if reducedDim else ''}"):
			shutil.rmtree(f"data/CTDDDAs_{dataType}{'-reducedDim' if reducedDim else ''}")

		for directory in [
			f"data/CTDDDAs_{dataType}{'-reducedDim' if reducedDim else ''}", 
			f"data/CTDDDAs_{dataType}{'-reducedDim' if reducedDim else ''}/marker", 
			f"data/CTDDDAs_{dataType}{'-reducedDim' if reducedDim else ''}/therapeutic"
		]:
			os.mkdir(directory)

	with open("files/CTD_DDAs.txt") as vectorFile:
		numMarkers = 0
		numTherapeutic = 0
		for line in vectorFile:
			info = line.split(" ; ")
			ids = info[0].split("#")
			label = "marker" if info[1] == "marker/mechanism" else "therapeutic"
			chemicalId = ids[0].split(":")[2]
			diseaseId = ids[1].split(":")[2]
			
			associationId = f"{chemicalId}_{diseaseId}"

			if not os.path.isfile(f"vectors/vectors2019{'-reducedDim' if reducedDim else ''}/chemical/{chemicalId}.npy") \
			or not os.path.isfile(f"vectors/vectors2019{'-reducedDim' if reducedDim else ''}/disease/{diseaseId}.npy"):
				continue

			chemicalVector = np.load(f"vectors/vectors2019{'-reducedDim' if reducedDim else ''}/chemical/{chemicalId}.npy")
			diseaseVector = np.load(f"vectors/vectors2019{'-reducedDim' if reducedDim else ''}/disease/{diseaseId}.npy")

			for dataType in dataTypes:
				associationVector = generateAssociationVector(chemicalVector, diseaseVector, dataType)
				np.save(f"data/CTDDDAs_{dataType}{'-reducedDim' if reducedDim else ''}/{label}/{associationId}.npy", associationVector)

			if label == "marker":
				numMarkers += 1
			else:
				numTherapeutic += 1

		vectorFile.close()

		print(f"{numMarkers} marker associations")
		print(f"{numTherapeutic} therapeutic associations")

def compileCTDDDAsWithSemantics(dataType='sparse'):
	if not os.path.exists("data"):
		os.mkdir("data")
	
	if os.path.exists(f"data/CTDDDAsWith{'Dense' if dense else 'Sparse'}Semantics"):
		shutil.rmtree(f"data/CTDDDAsWith{'Dense' if dense else 'Sparse'}Semantics")

	for directory in [
		f"data/CTDDDAsWith{'Dense' if dense else 'Sparse'}Semantics", 
		f"data/CTDDDAsWith{'Dense' if dense else 'Sparse'}Semantics/marker", 
		f"data/CTDDDAsWith{'Dense' if dense else 'Sparse'}Semantics/therapeutic"
	]:
		os.mkdir(directory)

	with open("files/CTD_DDAs.txt") as vectorFile:
		numMarkers = 0
		numTherapeutic = 0
		for line in vectorFile:
			info = line.split(" ; ")
			ids = info[0].split("#")
			label = "marker" if info[1] == "marker/mechanism" else "therapeutic"
			chemicalId = ids[0].split(":")[2]
			diseaseId = ids[1].split(":")[2]
			
			associationId = f"{chemicalId}_{diseaseId}"

			if not os.path.isfile(f"vector/vectors2019/chemical/{chemicalId}.npy") \
			or not os.path.isfile(f"vector/vectors2019/disease/{diseaseId}.npy") \
			or not os.path.isfile(f"vector/{'dense' if dense else 'sparse'}MeSHVectors/chemical/{chemicalId}.npy") \
			or not os.path.isfile(f"vector/{'dense' if dense else 'sparse'}MeSHVectors/disease/{diseaseId}.npy") \
			or not os.path.isfile(f"vector/{'dense' if dense else 'sparse'}ATCVectors/{chemicalId}.npy"):
				continue

			chemicalVector = np.load(f"vectors/vectors2019/chemical/{chemicalId}.npy")
			diseaseVector = np.load(f"vectors/vectors2019/disease/{diseaseId}.npy")
			chemicalMeSHVector = np.load(f"vectors/{'dense' if dense else 'sparse'}MeSHVectors/chemical/{chemicalId}.npy")
			chemicalATCVector = np.load(f"vectors/{'dense' if dense else 'sparse'}ATCVectors/{chemicalId}.npy")
			diseaseMeSHVector = np.load(f"vectors/{'dense' if dense else 'sparse'}MeSHVectors/disease/{diseaseId}.npy")

			associationVector = np.concatenate([chemicalVector, diseaseVector, chemicalMeSHVector, chemicalATCVector, diseaseMeSHVector])
			np.save(f"data/CTDDDAsWith{'Dense' if dense else 'Sparse'}Semantics/{label}/{associationId}.npy", associationVector)

			if label == "marker":
				numMarkers += 1
			else:
				numTherapeutic += 1

		vectorFile.close()

		print(f"{numMarkers} marker associations")
		print(f"{numTherapeutic} therapeutic associations")

def compilePredictedDDAs(dataTypes=[STANDARD_DATA_TYPE]):
	markerIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/marker')]
	therapeuticIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/therapeutic')]

	if not os.path.exists('data'):
		os.mkdir('data')

	for NDNs in ['5', '10', '20', '50', '100', 'Adaptive']:	
		print(f"{NDNs} NDNs")

		for dataType in dataTypes:
			if os.path.exists(f"data/predictedDDAs-{NDNs}NDNs_{dataType}"):
				shutil.rmtree(f"data/predictedDDAs-{NDNs}NDNs_{dataType}")
	
			for directory in [
				f"data/predictedDDAs-{NDNs}NDNs_{dataType}",
				f"data/predictedDDAs-{NDNs}NDNs_{dataType}/therapeutic",
				f"data/predictedDDAs-{NDNs}NDNs_{dataType}/marker"
			]:
				os.mkdir(directory)

		with open(f"files/predictedDDAs/allPredictedDDAs_2019_{NDNs}-NDNs-per-drug.txt", 'r') as vectorFile:
			numMarkers = 0
			numTherapeutic = 0
			labelsMissing = 0
			for line in vectorFile:
				info = line.split(";")
				ids = info[0].split("#")
				chemicalId = ids[0].split(":")[2]
				diseaseId = ids[1].split(":")[2]
				
				associationId = f"{chemicalId}_{diseaseId}"
				
				if associationId in markerIds:
					numMarkers += 1
					label = 'marker'
				elif associationId in therapeuticIds:
					numTherapeutic += 1
					label = 'therapeutic'
				else:
					labelsMissing += 1
					continue

				if not os.path.isfile(f"vectors/vectors2019/chemical/{chemicalId}.npy") \
				or not os.path.isfile(f"vectors/vectors2019/disease/{diseaseId}.npy"):
					continue

				chemicalVector = np.load(f"vectors/vectors2019/chemical/{chemicalId}.npy")
				diseaseVector = np.load(f"vectors/vectors2019/disease/{diseaseId}.npy")

				for dataType in dataTypes:
					associationVector = generateAssociationVector(chemicalVector, diseaseVector, dataType)
					np.save(f"data/predictedDDAs-{NDNs}NDNs_{dataType}/{label}/{associationId}.npy", associationVector)

			print(f"\t{numMarkers} marker associations")
			print(f"\t{numTherapeutic} therapeutic associations")
			print(f"\t{labelsMissing} labels missing")


def compilePredictedDDAsWithSemantics(dataType='full'):
	markerIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/marker')]
	therapeuticIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/therapeutic')]
	
	if not os.path.exists('data'):
		os.mkdir('data')

	for NDNs in ['5', '10', '20', '50', '100', 'Adaptive']:
		print(f"{NDNs} NDNs")
		
		if os.path.exists(f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}"):
			shutil.rmtree(f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}")
			
		for directory in [
						  f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}",
						  f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/therapeutic",
						  f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/marker"
						  ]:
			os.mkdir(directory)

		with open(f"files/predictedDDAs/allPredictedDDAs_2019_{NDNs}-NDNs-per-drug.txt") as vectorFile:
			numMarkers = 0
			numTherapeutic = 0
			for line in vectorFile:
				info = line.split(";")
				ids = info[0].split("#")
				chemicalId = ids[0].split(":")[2]
				diseaseId = ids[1].split(":")[2]
				
				associationId = f"{chemicalId}_{diseaseId}"
				
				if associationId in markerIds:
					label = 'marker'
				elif associationId in therapeuticIds:
					label = 'therapeutic'
				else:
					continue
			
				chemicalATCFiles = [file for file in os.listdir('vectors/ATCVectors') if file.startswith(chemicalId)]
				chemicalMeSHFiles = [file for file in os.listdir('vectors/MeSHVectors/chemical') if file.startswith(chemicalId)]
				diseaseMeSHFiles = [file for file in os.listdir('vectors/MeSHVectors/disease') if file.startswith(diseaseId)]
				
				if not os.path.isfile(f"vectors/vectors2019/chemical/{chemicalId}.npy") or not os.path.isfile(f"vectors/vectors2019/disease/{diseaseId}.npy"):
					continue
				
				chemicalVector = np.load(f"vectors/vectors2019/chemical/{chemicalId}.npy")
				diseaseVector = np.load(f"vectors/vectors2019/disease/{diseaseId}.npy")

				if os.path.exists(f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/{label}/{associationId}"):
					continue
				os.mkdir(f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/{label}/{associationId}")
				
				i = 0
				for chemicalMeSHFile in chemicalMeSHFiles:
					for chemicalATCFile in chemicalATCFiles:
						for diseaseMeSHFile in diseaseMeSHFiles:
							chemicalMeSHVector = np.load(f'vectors/MeSHVectors/chemical/{chemicalMeSHFile}')
							chemicalATCVector = np.load(f'vectors/ATCVectors/{chemicalATCFile}')
							diseaseMeSHVector = np.load(f'vectors/MeSHVectors/disease/{diseaseMeSHFile}')
							
							associationVector = np.concatenate([chemicalVector, diseaseVector, chemicalMeSHVector, chemicalATCVector, diseaseMeSHVector])
							
							np.save(f"data/predictedDDAsWithSemantics-{NDNs}NDNs_{dataType}/{label}/{associationId}/{i}.npy", associationVector)
							
							i += 1
							
							if label == 'marker':
								numMarkers += 1
							else:
								numTherapeutic += 1

			print(f"\t{numMarkers} marker associations")
			print(f"\t{numTherapeutic} therapeutic associations")

def compileRetrospectiveAnalysisDataset(ndnNums = [5, 10, 20, 50, 100, 'Adaptive'], dataTypes=[STANDARD_DATA_TYPE], reducedDim=False):
	if not os.path.exists('data'):
		os.mkdir('data')

	markerIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/marker')]
	therapeuticIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/therapeutic')]

	for NDNs in ndnNums:
		print(f'{NDNs} NDNs')
		for dataType in dataTypes:
			if os.path.exists(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}{'-reducedDim' if reducedDim else ''}"):
				shutil.rmtree(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}{'-reducedDim' if reducedDim else ''}")

			for path in [
				f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}{'-reducedDim' if reducedDim else ''}",
				f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}{'-reducedDim' if reducedDim else ''}/test",
				f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}{'-reducedDim' if reducedDim else ''}/test/marker",
				f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}{'-reducedDim' if reducedDim else ''}/test/therapeutic",
				f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}{'-reducedDim' if reducedDim else ''}/train",
				f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}{'-reducedDim' if reducedDim else ''}/train/marker",
				f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}{'-reducedDim' if reducedDim else ''}/train/therapeutic"
			]:
				os.mkdir(path)

		numMarkersTrain = 0
		numTherapeuticTrain = 0

		numMarkersTest = 0
		numTherapeuticTest = 0

		with open(f"files/predictedDDAs/allPredictedDDAs_1989_{NDNs}-NDNs-per-drug.txt", 'r') as associationFile:
			for line in associationFile:
				info = line.split(";")
				ids = info[0].split("#")
				chemicalId = ids[0].split(":")[2]
				diseaseId = ids[1].split(":")[2]

				associationId = f"{chemicalId}_{diseaseId}"

				test = info[1] == 'predicted'

				if associationId in markerIds:
					label = 'marker'
				elif associationId in therapeuticIds:
					label = 'therapeutic'
				else:
					continue

				if not os.path.isfile(f"vectors/vectors1989{'-reducedDim' if reducedDim else ''}/chemical/{chemicalId}.npy") or not os.path.isfile(f"vectors/vectors1989{'-reducedDim' if reducedDim else ''}/disease/{diseaseId}.npy"):
					continue

				chemicalVector = np.load(f"vectors/vectors2019{'-reducedDim' if reducedDim else ''}/chemical/{chemicalId}.npy")
				diseaseVector = np.load(f"vectors/vectors2019{'-reducedDim' if reducedDim else ''}/disease/{diseaseId}.npy")

				for dataType in dataTypes:
					associationVector = generateAssociationVector(chemicalVector, diseaseVector, dataType)
					np.save(f"data/retrospectiveAnalysis-{NDNs}NDNs_{dataType}{'-reducedDim' if reducedDim else ''}/{'test' if test else 'train'}/{label}/{associationId}.npy", associationVector)

				if label == 'marker':
					if test:
						numMarkersTest += 1
					else:
						numMarkersTrain += 1
				if label == 'therapeutic':
					if test:
						numTherapeuticTest += 1
					else:
						numTherapeuticTrain += 1


		print('\ttest')
		print(f'\t\t{numMarkersTest} marker')
		print(f'\t\t{numTherapeuticTest} therapeutic')

		print('\ttrain')
		print(f'\t\t{numMarkersTrain} marker')
		print(f'\t\t{numTherapeuticTrain} therapeutic')

def compileRetrospectiveAnalysisDatasetWithDistance():
	if not os.path.exists('data'):
		os.mkdir('data')

	markerIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/marker')]
	therapeuticIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/therapeutic')]

	for NDNs in [5, 10, 20, 50, 100, 'Adaptive']:
		print(f'{NDNs} NDNs')
		if os.path.exists(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs"):
			shutil.rmtree(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs")

		for path in [
			f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs",
			f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/test",
			f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/test/marker",
			f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/test/therapeutic",
			f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/train",
			f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/train/marker",
			f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/train/therapeutic"
		]:
			os.mkdir(path)

		numMarkersTrain = 0
		numTherapeuticTrain = 0

		numMarkersTest = 0
		numTherapeuticTest = 0

		with open(f"files/predictedDDAs/allPredictedDDAs_1989_{NDNs}-NDNs-per-drug.txt", 'r') as associationFile:
			for line in associationFile:
				info = line.split(";")
				ids = info[0].split("#")
				chemicalId = ids[0].split(":")[2]
				diseaseId = ids[1].split(":")[2]

				associationId = f"{chemicalId}_{diseaseId}"

				test = info[1] == 'predicted'

				if associationId in markerIds:
					if test:
						numMarkersTest += 1
					else:
						numMarkersTrain += 1
					label = 'marker'
				elif associationId in therapeuticIds:
					if test:
						numTherapeuticTest += 1
					else:
						numTherapeuticTrain += 1
					label = 'therapeutic'
				else:
					continue

				if not os.path.isfile(f"vectors/vectors1989/chemical/{chemicalId}.npy") or not os.path.isfile(f"vectors1989/disease/{diseaseId}.npy"):
					continue

				chemicalVector = np.load(f"vectors/vectors1989/chemical/{chemicalId}.npy")
				diseaseVector = np.load(f"vectors/vectors1989/disease/{diseaseId}.npy")

				distance = float(info[3])

				associationVector = np.concatenate([chemicalVector, diseaseVector, [distance]])
				np.save(f"data/retrospectiveAnalysisWithDistance-{NDNs}NDNs/{'test' if test else 'train'}/{label}/{associationId}.npy", associationVector)


		print('\ttest')
		print(f'\t\t{numMarkersTest} marker')
		print(f'\t\t{numTherapeuticTest} therapeutic')

		print('\ttrain')
		print(f'\t\t{numMarkersTrain} marker')
		print(f'\t\t{numTherapeuticTrain} therapeutic')

#def compileRetrospectiveAnalysisDatasetWithSemantics(dataType='sparse'):
#	if not os.path.exists('data'):
#		os.mkdir('data')
#
#	markerIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/marker')]
#	therapeuticIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/therapeutic')]
#
#	for NDNs in [5, 10, 20, 50, 100, 'Adaptive']:
#		print(f'{NDNs} NDNs')
#
#		if os.path.exists(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}"):
#			shutil.rmtree(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}")
#
#		for path in [
#			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}",
#			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test",
#			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/marker",
#			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/therapeutic",
#			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train",
#			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/marker",
#			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/therapeutic"
#		]:
#			os.mkdir(path)
#
#		numMarkersTrain = 0
#		numTherapeuticTrain = 0
#
#		numMarkersTest = 0
#		numTherapeuticTest = 0
#
#		with open(f"files/predictedDDAs/allPredictedDDAs_1989_{NDNs}-NDNs-per-drug.txt", 'r') as associationFile:
#			for line in associationFile:
#				info = line.split(";")
#				ids = info[0].split("#")
#				chemicalId = ids[0].split(":")[2]
#				diseaseId = ids[1].split(":")[2]
#
#				associationId = f"{chemicalId}_{diseaseId}"
#
#				test = info[1] == 'predicted'
#
#				if not os.path.isfile(f"vectors/vectors1989/chemical/{chemicalId}.npy") \
#				or not os.path.isfile(f"vectors/vectors1989/disease/{diseaseId}.npy") \
#				or not os.path.isfile(f'vectors/{dataType}MeSHVectors/chemical/{chemicalId}.npy') \
#				or not os.path.isfile(f'vectors/{dataType}MeSHVectors/disease/{diseaseId}.npy') \
#				or not os.path.isfile(f'vectors/{dataType}ATCVectors/{chemicalId}.npy'):
#					continue
#
#				if associationId in markerIds:
#					if test:
#						numMarkersTest += 1
#					else:
#						numMarkersTrain += 1
#					label = 'marker'
#				elif associationId in therapeuticIds:
#					if test:
#						numTherapeuticTest += 1
#					else:
#						numTherapeuticTrain += 1
#					label = 'therapeutic'
#				else:
#					continue
#
#				chemicalVector = np.load(f"vectors/vectors1989/chemical/{chemicalId}.npy")
#				diseaseVector = np.load(f"vectors/vectors1989/disease/{diseaseId}.npy")
#				chemicalMeSHVector = np.load(f'vectors/{dataType}MeSHVectors/chemical/{chemicalId}.npy')
#				chemicalATCVector = np.load(f'vectors/{dataType}ATCVectors/{chemicalId}.npy')
#				diseaseMeSHVector = np.load(f'vectors/{dataType}MeSHVectors/disease/{diseaseId}.npy')
#
#				associationVector = np.concatenate([chemicalVector, diseaseVector, chemicalMeSHVector, chemicalATCVector, diseaseMeSHVector])
#				np.save(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/{'test' if test else 'train'}/{label}/{associationId}.npy", \
#					associationVector)
#
#
#		print('\ttest')
#		print(f'\t\t{numMarkersTest} marker')
#		print(f'\t\t{numTherapeuticTest} therapeutic')
#
#		print('\ttrain')
#		print(f'\t\t{numMarkersTrain} marker')
#		print(f'\t\t{numTherapeuticTrain} therapeutic')

def compileWeightedRetrospectiveAnalysisDataset(dataTypes=[STANDARD_DATA_TYPE]):
	if not os.path.exists('data'):
		os.mkdir('data')

	markerIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/marker')]
	therapeuticIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/therapeutic')]

	numPMIDs = {}

	with open('files/existingDDAs/allExisitingDDAsInTextWithPMIDS_minCount_1_till_1989.txt', 'r') as associationFile:
		for line in associationFile:
			info = line.split(";")
			ids = info[0].split("#")
			try:
				chemicalId = ids[0].split(":")[2]
				diseaseId = ids[1].split(":")[2]
			except IndexError:
				continue
			associationId = f"{chemicalId}_{diseaseId}"
			pmids = info[1].split(' ')
			numPMIDs[associationId] = len(pmids)

	for NDNs in [5, 10, 20, 50, 100, 'Adaptive']:
		print(f'{NDNs} NDNs')
		for dataType in dataTypes:
			if os.path.exists(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}"):
				shutil.rmtree(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}")

			for path in [
				f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}",
				f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/test",
				f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/test/marker",
				f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/test/therapeutic",
				f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/train",
				f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/train/marker",
				f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/train/therapeutic"
			]:
				os.mkdir(path)

		numMarkersTrain = 0
		numTherapeuticTrain = 0

		numMarkersTest = 0
		numTherapeuticTest = 0

		with open(f"files/predictedDDAs/allPredictedDDAs_1989_{NDNs}-NDNs-per-drug.txt", 'r') as associationFile:
			for line in associationFile:
				info = line.split(";")
				ids = info[0].split("#")
				chemicalId = ids[0].split(":")[2]
				diseaseId = ids[1].split(":")[2]

				associationId = f"{chemicalId}_{diseaseId}"

				test = info[1] == 'predicted'

				weight = 1

				if associationId in numPMIDs.keys():
					if numPMIDs[associationId] > 100:
						weight = 5
					elif numPMIDs[associationId] > 50:
						weight = 4
					elif numPMIDs[associationId] > 20:
						weight = 3
					elif numPMIDs[associationId] > 10:
						weight = 2

				if associationId in markerIds:
					if test:
						numMarkersTest += weight
					else:
						numMarkersTrain += weight
					label = 'marker'
				elif associationId in therapeuticIds:
					if test:
						numTherapeuticTest += weight
					else:
						numTherapeuticTrain += weight
					label = 'therapeutic'
				else:
					continue

				if not os.path.isfile(f"vectors/vectors1989/chemical/{chemicalId}.npy") or not os.path.isfile(f"vectors/vectors1989/disease/{diseaseId}.npy"):
					continue

				chemicalVector = np.load(f"vectors/vectors1989/chemical/{chemicalId}.npy")
				diseaseVector = np.load(f"vectors/vectors1989/disease/{diseaseId}.npy")

				for i in range(weight):
					for dataType in dataTypes:
						associationVector = generateAssociationVector(chemicalVector, diseaseVector, dataType)
						np.save(f"data/weightedRetrospectiveAnalysis-{NDNs}NDNs_{dataType}/{'test' if test else 'train'}/{label}/{associationId}-{i}.npy", associationVector)


		print('\ttest')
		print(f'\t\t{numMarkersTest} marker')
		print(f'\t\t{numTherapeuticTest} therapeutic')

		print('\ttrain')
		print(f'\t\t{numMarkersTrain} marker')
		print(f'\t\t{numTherapeuticTrain} therapeutic')

def compileSemanticVectors():
	if not os.path.exists('vectors'):
		os.mkdir('vectors')

	if os.path.exists('vectors/MeSHVectors'):
		shutil.rmtree('vectors/MeSHVectors')

	if os.path.exists('vectors/ATCVectors'):
		shutil.rmtree('vectors/ATCVectors')

	for path in ['vectors/ATCVectors', 'vectors/MeSHVectors', 'vectors/MeSHVectors/chemical', 'vectors/MeSHVectors/disease']:
		os.mkdir(path)

	maxTreeLenMeSHDisease = 0
	maxTreeLenMeSHDrug = 0
	chars = set()
	with open('files/diseaseMeshTrees.txt', 'r') as file:
		for line in file:
			id = line.split('#')[0].split(':')[1]
			trees = [tree.split('/')[0].split('.') for tree in line.split('#')[1].strip('\n').strip('[').strip(']').split(', ')]

			for tree in trees:
				if len(tree) > maxTreeLenMeSHDisease:
					maxTreeLenMeSHDisease = len(tree)
				chars.update([char for char in tree])

	with open('files/drugMeshTreesATCandAHFS.txt', 'r') as file:
		for line in file:
			id = line.split('#')[0].split(':')[1]
			trees = [tree.split('/')[0].split('.') for tree in line.split('#')[1].strip('\n').strip('[').strip(']').split(', ')]

			for tree in trees:
				if len(tree) > maxTreeLenMeSHDrug:
					maxTreeLenMeSHDrug = len(tree)
				chars.update([char for char in tree])

	with open('files/drugMeshTreesATCandAHFS.txt', 'r') as file:
		for line in file:
			trees = line.split('#')[2].strip('\n').strip('[').strip(']').split(', ')

			for tree in trees:
				chars.update([char for char in tree])

	chars = list(chars)
	chars = dict([(chars[i], i + 1) for i in range(len(chars))])

	print(f'Num chars: {len(chars)}')

	numMeSHDrugVectors = 0
	numMeSHDiseaseVectors = 0
	numATCVectors = 0

	with open('files/diseaseMeshTrees.txt', 'r') as file:
		for line in file:
			id = line.split('#')[0].split(':')[1]
			trees = [tree.split('/')[0].split('.') for tree in line.split('#')[1].strip('\n').strip('[').strip(']').split(', ')]

			for i in range(len(trees)): 
				vector = np.array([chars[char] for char in trees[i]] + [0] * (maxTreeLenMeSHDisease - len(trees[i])))
				np.save(f'vectors/MeSHVectors/disease/{id}_{i}.npy', vector)

				numMeSHDiseaseVectors += 1

	with open('files/drugMeshTreesATCandAHFS.txt', 'r') as file:
		for line in file:
			id = line.split('#')[0].split(':')[1]
			trees = [tree.split('/')[0].split('.') for tree in line.split('#')[1].strip('\n').strip('[').strip(']').split(', ')]

			for i in range(len(trees)):
				if len(trees[i]) == 0:
					continue
					
				vector = np.array([chars[char] for char in trees[i]] + [0] * (maxTreeLenMeSHDrug - len(trees[i])))
				np.save(f'vectors/MeSHVectors/chemical/{id}_{i}.npy', vector)

				numMeSHDrugVectors += 1

	with open('files/drugMeshTreesATCandAHFS.txt', 'r') as file:
		for line in file:
			id = line.split('#')[0].split(':')[1]
			trees = line.split('#')[2].strip('\n').strip('[').strip(']').split(', ')

			for i in range(len(trees)):
				if len(trees[i]) != 7:
					continue

				vector = np.array([chars[char] for char in trees[i]])
				np.save(f'vectors/ATCVectors/{id}_{i}.npy', vector)

				numATCVectors += 1

	chemicalMeSHDummy = np.array([0] * 33)
	chemicalATCDummy = np.array([0] * 7)
	diseaseMeSHDummy = np.array([0] * 30)

	np.save("vectors/MeSHVectors/chemical/zeros.npy", chemicalMeSHDummy)
	np.save("vectors/ATCVectors/zeros.npy", chemicalATCDummy)
	np.save("vectors/MeSHVectors/disease/zeros.npy", diseaseMeSHDummy)

	print(f'{numMeSHDiseaseVectors} MeSH disease Vectors')
	print(f'{numMeSHDrugVectors} MeSH drug Vectors')
	print(f'{numATCVectors} ATC Vectors')

def compileSemanticCTDDDAs():
	if not os.path.exists("data"):
		os.mkdir("data")
	
	if os.path.exists("data/semanticCTDDDAs"):
		shutil.rmtree("data/semanticCTDDDAs")

	for directory in [
		"data/semanticCTDDDAs", 
		"data/semanticCTDDDAs/marker", 
		"data/semanticCTDDDAs/therapeutic"
	]:
		os.mkdir(directory)

	with open("files/CTD_DDAs.txt") as vectorFile:
		numMarkers = 0
		numTherapeutic = 0
		for line in vectorFile:
			info = line.split(" ; ")
			ids = info[0].split("#")
			label = "marker" if info[1] == "marker/mechanism" else "therapeutic"
			chemicalId = ids[0].split(":")[2]
			diseaseId = ids[1].split(":")[2]
			
			associationId = f"{chemicalId}_{diseaseId}"

			chemicalATCFiles = [file for file in os.listdir('vectors/ATCVectors') if file.startswith(chemicalId)]
			chemicalMeSHFiles = [file for file in os.listdir('vectors/MeSHVectors/chemical') if file.startswith(chemicalId)]
			diseaseMeSHFiles = [file for file in os.listdir('vectors/MeSHVectors/disease') if file.startswith(diseaseId)]

			os.mkdir(f"data/semanticCTDDDAs/{label}/{associationId}")

			i = 0
			for chemicalMeSHFile in chemicalMeSHFiles:
				for chemicalATCFile in chemicalATCFiles:
					for diseaseMeSHFile in diseaseMeSHFiles:
						chemicalMeSHVector = np.load(f'vectors/MeSHVectors/chemical/{chemicalMeSHFile}')
						chemicalATCVector = np.load(f'vectors/ATCVectors/{chemicalATCFile}')
						diseaseMeSHVector = np.load(f'vectors/MeSHVectors/disease/{diseaseMeSHFile}')

						associationVector = np.concatenate([chemicalMeSHVector, chemicalATCVector, diseaseMeSHVector])

						np.save(f"data/semanticCTDDDAs/{label}/{associationId}/{i}.npy", associationVector)

						i += 1

						if label == "marker":
							numMarkers += 1
						else:
							numTherapeutic += 1

		vectorFile.close()

		print(f"{numMarkers} marker associations")
		print(f"{numTherapeutic} therapeutic associations")

def compileSemanticCTDDDAsWithPubmed():
	if not os.path.exists("data"):
		os.mkdir("data")
	
	if os.path.exists("data/semanticCTDDDAsWithPubmed"):
		shutil.rmtree("data/semanticCTDDDAsWithPubmed")

	for directory in [
		"data/semanticCTDDDAsWithPubmed", 
		"data/semanticCTDDDAsWithPubmed/marker", 
		"data/semanticCTDDDAsWithPubmed/therapeutic"
	]:
		os.mkdir(directory)

	with open("files/CTD_DDAs.txt") as vectorFile:
		numMarkers = 0
		numTherapeutic = 0
		for line in vectorFile:
			info = line.split(" ; ")
			ids = info[0].split("#")
			label = "marker" if info[1] == "marker/mechanism" else "therapeutic"
			chemicalId = ids[0].split(":")[2]
			diseaseId = ids[1].split(":")[2]
			
			associationId = f"{chemicalId}_{diseaseId}"

			if not os.path.isfile(f"vectors/vectors2019/chemical/{chemicalId}.npy") or not os.path.isfile(f"vectors/vectors2019/disease/{diseaseId}.npy"):
				continue

			chemicalVector = np.load(f"vectors/vectors2019/chemical/{chemicalId}.npy")
			diseaseVector = np.load(f"vectors/vectors2019/disease/{diseaseId}.npy")

			chemicalATCFiles = [file for file in os.listdir('vectors/ATCVectors') if file.startswith(chemicalId)]
			chemicalMeSHFiles = [file for file in os.listdir('vectors/MeSHVectors/chemical') if file.startswith(chemicalId)]
			diseaseMeSHFiles = [file for file in os.listdir('vectors/MeSHVectors/disease') if file.startswith(diseaseId)]

			os.mkdir(f"data/semanticCTDDDAsWithPubmed/{label}/{associationId}")

			i = 0
			for chemicalMeSHFile in chemicalMeSHFiles:
				for chemicalATCFile in chemicalATCFiles:
					for diseaseMeSHFile in diseaseMeSHFiles:
						chemicalMeSHVector = np.load(f'vectors/MeSHVectors/chemical/{chemicalMeSHFile}')
						chemicalATCVector = np.load(f'vectors/ATCVectors/{chemicalATCFile}')
						diseaseMeSHVector = np.load(f'vectors/MeSHVectors/disease/{diseaseMeSHFile}')

						associationVector = np.concatenate([chemicalVector, diseaseVector, chemicalMeSHVector, chemicalATCVector, diseaseMeSHVector])

						np.save(f"data/semanticCTDDDAsWithPubmed/{label}/{associationId}/{i}.npy", associationVector)

						i += 1

						if label == "marker":
							numMarkers += 1
						else:
							numTherapeutic += 1

		vectorFile.close()

		print(f"{numMarkers} marker associations")
		print(f"{numTherapeutic} therapeutic associations")

def compileSemanticRetrospectiveAnalysisDataset():
	if not os.path.exists('data'):
		os.mkdir('data')

	markerIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/marker')]
	therapeuticIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/therapeutic')]

	for NDNs in [5, 10, 20, 50, 100, 'Adaptive']:
		print(f'{NDNs} NDNs')

		if os.path.exists(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs"):
			shutil.rmtree(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs")

		for path in [
			f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs",
			f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/test",
			f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/test/marker",
			f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/test/therapeutic",
			f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/train",
			f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/train/marker",
			f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/train/therapeutic"
		]:
			os.mkdir(path)

		numMarkersTrain = 0
		numTherapeuticTrain = 0

		numMarkersTest = 0
		numTherapeuticTest = 0

		with open(f"files/predictedDDAs/allPredictedDDAs_1989_{NDNs}-NDNs-per-drug.txt", 'r') as associationFile:
			for line in associationFile:
				info = line.split(";")
				ids = info[0].split("#")
				chemicalId = ids[0].split(":")[2]
				diseaseId = ids[1].split(":")[2]

				associationId = f"{chemicalId}_{diseaseId}"

				test = info[1] == 'predicted'

				if associationId in markerIds:
					label = 'marker'
				elif associationId in therapeuticIds:
					label = 'therapeutic'
				else:
					continue
				
				chemicalATCFiles = [file for file in os.listdir('vectors/ATCVectors') if file.startswith(chemicalId)]
				chemicalMeSHFiles = [file for file in os.listdir('vectors/MeSHVectors/chemical') if file.startswith(chemicalId)]
				diseaseMeSHFiles = [file for file in os.listdir('vectors/MeSHVectors/disease') if file.startswith(diseaseId)]

				i = 0
				for chemicalMeSHFile in chemicalMeSHFiles:
					for chemicalATCFile in chemicalATCFiles:
						for diseaseMeSHFile in diseaseMeSHFiles:
							chemicalMeSHVector = np.load(f'vectors/MeSHVectors/chemical/{chemicalMeSHFile}')
							chemicalATCVector = np.load(f'vectors/ATCVectors/{chemicalATCFile}')
							diseaseMeSHVector = np.load(f'vectors/MeSHVectors/disease/{diseaseMeSHFile}')

							associationVector = np.concatenate([chemicalMeSHVector, chemicalATCVector, diseaseMeSHVector])

							np.save(f"data/semanticRetrospectiveAnalysis-{NDNs}NDNs/{'test' if test else 'train'}/{label}/{associationId}_{i}.npy", associationVector)

							i += 1

							if label == 'marker':
								if test:
									numMarkersTest += 1
								else:
									numMarkersTrain += 1
							else:
								if test:
									numTherapeuticTest += 1
								else:
									numTherapeuticTrain += 1

		print('\ttest')
		print(f'\t\t{numMarkersTest} marker')
		print(f'\t\t{numTherapeuticTest} therapeutic')

		print('\ttrain')
		print(f'\t\t{numMarkersTrain} marker')
		print(f'\t\t{numTherapeuticTrain} therapeutic')

def compileRetrospectiveAnalysisDatasetWithSemantics(dataType='full'):
	if not os.path.exists('data'):
		os.mkdir('data')

	markerIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/marker')]
	therapeuticIds = [filename.strip('.npy') for filename in os.listdir('data/CTDDDAs_concat/therapeutic')]

	for NDNs in [5, 10, 20, 50, 100, 'Adaptive']:
		print(f'{NDNs} NDNs')
		
		if os.path.exists(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}"):
			shutil.rmtree(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}")

		for path in [
			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}",
			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test",
			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/marker",
			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/test/therapeutic",
			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train",
			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/marker",
			f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/train/therapeutic"
		]:
			os.mkdir(path)

		numMarkersTrain = 0
		numTherapeuticTrain = 0

		numMarkersTest = 0
		numTherapeuticTest = 0

		with open(f"files/predictedDDAs/allPredictedDDAs_1989_{NDNs}-NDNs-per-drug.txt", 'r') as associationFile:
			for line in associationFile:
				info = line.split(";")
				ids = info[0].split("#")
				chemicalId = ids[0].split(":")[2]
				diseaseId = ids[1].split(":")[2]

				associationId = f"{chemicalId}_{diseaseId}"

				test = info[1] == 'predicted'

				if associationId in markerIds:
					label = 'marker'
				elif associationId in therapeuticIds:
					label = 'therapeutic'
				else:
					continue
				
				chemicalATCFiles = [file for file in os.listdir('vectors/ATCVectors') if file.startswith(chemicalId)]
				chemicalMeSHFiles = [file for file in os.listdir('vectors/MeSHVectors/chemical') if file.startswith(chemicalId)]
				diseaseMeSHFiles = [file for file in os.listdir('vectors/MeSHVectors/disease') if file.startswith(diseaseId)]

				if not os.path.isfile(f"vectors/vectors1989/chemical/{chemicalId}.npy") or not os.path.isfile(f"vectors/vectors1989/disease/{diseaseId}.npy"):
					continue

				chemicalVector = np.load(f"vectors/vectors1989/chemical/{chemicalId}.npy")
				diseaseVector = np.load(f"vectors/vectors1989/disease/{diseaseId}.npy")

				i = 0
				for chemicalMeSHFile in chemicalMeSHFiles:
					for chemicalATCFile in chemicalATCFiles:
						for diseaseMeSHFile in diseaseMeSHFiles:
							chemicalMeSHVector = np.load(f'vectors/MeSHVectors/chemical/{chemicalMeSHFile}')
							chemicalATCVector = np.load(f'vectors/ATCVectors/{chemicalATCFile}')
							diseaseMeSHVector = np.load(f'vectors/MeSHVectors/disease/{diseaseMeSHFile}')

							associationVector = np.concatenate([chemicalVector, diseaseVector, chemicalMeSHVector, chemicalATCVector, diseaseMeSHVector])

							np.save(f"data/retrospectiveAnalysisWithSemantics-{NDNs}NDNs_{dataType}/{'test' if test else 'train'}/{label}/{associationId}_{i}.npy", associationVector)

							i += 1

							if label == 'marker':
								if test:
									numMarkersTest += 1
								else:
									numMarkersTrain += 1
							else:
								if test:
									numTherapeuticTest += 1
								else:
									numTherapeuticTrain += 1

		print('\ttest')
		print(f'\t\t{numMarkersTest} marker')
		print(f'\t\t{numTherapeuticTest} therapeutic')

		print('\ttrain')
		print(f'\t\t{numMarkersTrain} marker')
		print(f'\t\t{numTherapeuticTrain} therapeutic')
