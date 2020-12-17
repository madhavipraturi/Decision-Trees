from algorithm import ID3, CART, missClassification
import pandas as pd


if __name__ == '__main__':
	print('   ** DECISION TREE CLASSIFICATION **   ')
	algorithmUsed = input('Enter the method you want to use: (ID3: 1, CART: 2, MissClassification error: 3) ')
	if algorithmUsed == '1':
		algorithm = ID3()
	elif algorithmUsed == '2':
		algorithm = CART()
	elif algorithmUsed == '3':
		algorithm = missClassification()
	else:
		algorithm = CART()
		
	chiSquare = input('Chi Square Split stopping: (Yes/No) ')
	ci = 0
	if chiSquare == 'No' or chiSquare == 'no':
		chis = False
	else:
		chis = True
		confidenceInterval = input('Enter prefered confidence interval: (0,95,99)')
		if confidenceInterval == 0:
			chis = False
		elif ci == '95' or ci == '99':
			ci = int(confidenceInterval)
		else:
			ci = 99
		
	postP = input('Post Prune: (Yes/No) ')
	if postP == 'No' or postP == 'no':
		pp = False
	else:
		pp = True
	
	trainingData = 'trainingFinal.csv'
	testingData = 'testingFinal.csv'
	try:
		data1 = pd.read_csv(trainingData)
	except:
		print('WARNING: No training data found. Test data should be in "trainingData.csv" in the root folder.')
	try:
		data2 = pd.read_csv(testingData)
	except:
		print('WARNING: No test data found. Test data should be in "testingData.csv" in the root folder.')

	print('\nTraining...\n')
	algorithm.train(data1, postPrune = pp, chi = chis, confidence = ci)
	algorithm.predictData(data2)