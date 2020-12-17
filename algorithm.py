try:
	import pandas as pd
	import numpy as np
	from collections import Counter
	from predict import Predict
	import copy
	from graphviz import Digraph
	import random
except:
	x = input('WARNING: Requirements not installed. "pip install -r requirements.txt" to install.')


class DrawTree:
	# This class is for creating a visual output of the decision tree
	def __init__(self):
		try:
			self.graph = Digraph('output', filename='treeVisualization.gv', node_attr={'color': 'lightblue2', 'style': 'filled'}, format = 'png')
			self.graph.attr(size='4000,4000')
		except:
			pass

	def draw(self, tree):
		for i in tree.branches:
			self.graph.edge(tree.data, i.tree.data, label = i.branch)
			self.draw(i.tree)
		
	def visualizeTree(self, tree):
		self.draw(tree)
		self.graph.view()

class Branch:
	def __init__(self):
		self.branch = 'NONEL'
		self.tree = Tree()

class Tree:
	# This data structure stores the tree
	def __init__(self):
		self.data = 'NONE'
		self.ig = None
		self.probList = []
		self.branches = []
		
	def addNode(self, branch, tree):
		branchTemp = Branch()
		branchTemp.branch = branch
		branchTemp.tree = tree
		self.branches.append(branchTemp)
		
	def displayTree(self, tree):
		print(tree.data, tree.ig)
		for i in tree.branches:
			print('->', i.branch) 
			tree.displayTree(i.tree)
			

class Library:
	# The class Library contains all the common functions accessed by all the other classes
	def __init__(self):
		self.listAcc = []
		
	def getColumn(self, data, columnIndex):
		# Given a column index, returns the particular column from the dataframe as a list
		result = data.iloc[:, columnIndex].tolist()
		return result
	
	def findTotal(self, list):
		result = 0
		for i in list:
			result = result + i
		return result
	
	def splitData(self, data):
		# Randomly splits the data into train and test in a given ratio.
		value = int(data.shape[0]*3//4)
		testValue = data.shape[0] - value
		testList = []
		trainList = []
		for i in range(data.shape[0]):
			r = random.randint(0,2)
			if r == 0:
				if len(testList) < testValue:
					testList.append(i)
				else:
					trainList.append(i)
			else:
				trainList.append(i)
		
		train = data.iloc[trainList,:]
		test = data.iloc[testList,:]
		return train, test
				
# 			
# 	def splitData(self, data):
# 		# Split a dataset on a given ratio
# 		train = data.iloc[:int(data.shape[0]*3//4), :]
# 		test = data.iloc[int(data.shape[0]*3//4):, :]
# 		return train, test

	def getUniqueValues(self, list):
		# Returns all the unique values in a list
		values = Counter(list).values()
		result = []
		for i in values:
			result.append(i)
		return result
	
	def getUniqueKeys(self, list):
		values = Counter(list).keys()
		result = []
		for i in values:
			result.append(i)
		return result
	
	def getColumnName(self, dataFrame, columnIndex):
		return dataFrame.iloc[:, columnIndex].name
	
	def getResult(self, df):
		list = df.iloc[:, len(df.columns) - 1].tolist()
		return list[0]
	
	def stripClass(self, data):
		# Strips the result class from test data for calculating accuracy
		list = Library.getColumn(self, data, len(data.columns)-1)
		data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
		return data, list
	
	def highestProbabilty(self, list):
		max = 0 
		for i in list:
			if i[1] > max:
				max = i[1]
				node = i[0]
		return node
	
	def getProbabilityList(self, listOfOutcomes):
		# Given a list of values, returns the particular value and its probability of occuracnce in the list.
		keys = self.getUniqueKeys(listOfOutcomes)
		values = self.getUniqueValues(listOfOutcomes)
		
		result = []
		for i in range(len(keys)):
			result.append([keys[i], values[i]/sum(values)])
		return result
	

	def getPartialDataFrame(self, dataFrame, index, value):
		# Returns sub datasets given the selection criteria. 
		partialData = dataFrame.loc[dataFrame.iloc[:, index] == value]
		return partialData
	
	def replaceList(self, list, element1, element2):
		for i in list:
			if i == element1:
				i = element2
		return list
	
	def traverse(self, element, branch):
		if branch.tree.data == element:
			# Replace the tree removed with a node (Most probable outcome)
			node = self.highestProbabilty(branch.tree.probList)
			branch.tree.data = node
			branch.tree.branches = []
			return branch
		else:
			return None

	
	def removeElement(self, element, tree):
		# Removes a particular node from a given tree
		for i in tree.branches:
			branch = self.traverse(element, i)
			if branch != None:
				tree.branches = self.replaceList(tree.branches, i, branch)
			else:
				self.removeElement(element, i.tree)
		return tree
	
	def accuracyCompare(self, tree1, tree2):
		# Compare accuracy of two given trees
		acc1 = self.accuracy(tree1)
		acc2 = self.accuracy(tree2)
		if acc1 <= acc2:
			self.listAcc.append(acc2)
			return 1
		else:
			return 0
		
	def predictData(self, testData):
		# Method to specifically create a dataset with the resulting output for kaggle submission
		list = self.predict(testData, self.decisionTree)
		
		row = []
		index = 2001
		for i in list:
			row.append([index, i])
			index = index + 1
		
		print('out')
		finalOutput = pd.DataFrame(data = row, columns = ['id', 'class'])
		finalOutput.to_csv('predicted.csv', index=False)
		
	def getMostProbable(self, df):
		# Given a dataset, returns the most probable resulting class
		list = df.iloc[:, len(df.columns) - 1].tolist()
		pl = self.getProbabilityList(list)
		return self.highestProbabilty(pl)
	
	def getUniquelist(self, df1, df2):
		# Resturns a list with unique keys when given a list of values
		listA = self.getUniqueKeys(df1.iloc[:,0])
		listB = self.getUniqueKeys(df1.iloc[:,1])
		
		return listA, listB
	

	def calculateAccuracy(self, resultList, predictedList):
		# Computes percentage of correctly predicted values
		correct = 0
		for i in range(len(resultList)):
			try:
				if resultList[i] == predictedList[i]:
					correct = correct + 1
			except:
				pass
		return (correct/len(resultList)) * 100
	
	def predict(self, testData, tree):
		# Creates a object of Predict class and passes the test data to be predicted.
		pr = Predict(tree)
		result = pr.predict(testData)
		return result
	
	def getCriticalValue(self, dof, index):
		# Return critical value for chi square stopping test using the table
		if index == 95:
			i = 0
		elif index == 99:
			i = 1
			
		chiTable = {
			1 : [3.84, 6.63],
			3 : [7.82, 16.27],
			4 : [9.49, 13.28],
			6 : [12.59, 16.81],
			8 : [15.51, 20.09],
			12 : [21.03, 26.22],
			15 : [25, 30.58]}
		
		for key, value in chiTable.items():
			if key == dof:
				return value[i]
		return 0
	
	
	def getChiDataFrame(self, df, index):
		return df.iloc[:, [index, len(df.columns)-1]]
	
	def getCount(self, df, element1, element2):
		count = 0
		for i in range(len(df)):
			if df.iloc[i, 0] == element1 and df.iloc[i,1] == element2:
				count = count + 1
		return count
		

	def chiSquare(self, prev_df, df):
		#Computing chi square values
		if prev_df.empty == True:
			return 0, 0
		 
		listA, listB = self.getUniquelist(prev_df, df)
		actual = []
		expected = []

		# Composing a matrix of actual and expected count of each attribute (ex: A,G,T,C and N,EI,IE)
		for i in listA:
			actualRow = []
			expectedRow = []
			for j in listB:
				actualRow.append(self.getCount(prev_df, i, j))
				expectedRow.append(self.getCount(df, i, j))
			actual.append(actualRow)
			expected.append(expectedRow)

		chiSum = 0
		te = prev_df.shape[0] - 1
		ta = df.shape[0] - 1

		# Calculating chi square value using the matrix
		for i in range(len(actual)):
			for j in range(len(actual[0])):
				try:
					chi = (pow(2, actual[i][j] - expected[i][j]) / expected[i][j])
					self.accuracyList.append(chi)
				except:
					chi = 0
				chiSum = chiSum + chi
		
		dof = (len(actual)-1) * (len(actual[0])-1)
		return chiSum, dof
	
	def postPrune(self, tree):
		#Post pruning an existing tree
		for i in tree.branches:
			# When the leaf (end) of the tree is reached
			if i.tree.branches == []:
				if tree.data != self.decisionTree.data:
					treeFake = copy.deepcopy(self.decisionTree)
					treeCheck = self.removeElement(tree.data, treeFake)

					# Comparing accuracy of the tree where the node was removed and original tree.
					if Library.accuracyCompare(self, self.decisionTree, treeCheck) == 1:
						self.decisionTree = copy.deepcopy(treeCheck)
			else:
				self.postPrune(i.tree)
				
	def accuracy(self, tree):
		testData = self.testData
		resultList = self.resultList
		predictedList = self.predict(testData, tree)
		accuracy = self.calculateAccuracy(resultList, predictedList)
		return accuracy
	
	def train(self, trainingData, postPrune = False, chi = False, confidence = 95):
		self.accuracyFinal = 0
		self.accuracyList = []
		data1, data2 = self.splitData(trainingData)
		self.trainingData = data1
		self.testData, self.resultList = self.stripClass(data2)
		self.build(self.trainingData, postPrune, chi, confidence)
		print('\nAccuracy: ', self.accuracy(self.decisionTree))
		try:
			graph = DrawTree()
			graph.visualizeTree(self.decisionTree)
		except:
			print('WARNING: Add graphviz to PATH to render visual tree output. The predicted data can be found in "predicted.csv".')

	
class missClassification(Library):
	def __self__(self):
		self.trainingData = pd.DataFrame()
		self.decisionTree = Tree()
		self.counter = 0
		
	def calculateMiss(self, listOfOutcomes):
		values = Counter(listOfOutcomes).values()
			
		valueList = []
		for i in values:
			valueList.append(i)
		return self.missClassification(valueList)
	
	def missClassification(self, list):
		totalElements = Library.findTotal(self, list)
		probMax = 0
		for i in list:
			probOfElement = (i/totalElements)
			if probMax < probOfElement:
				probMax = probOfElement
		return 1 - probMax
	
	def getMinError(self, listOfGini):
		check = 0
		for i in listOfGini:
			if i[0] != 0.0:
				check = 1
		if check ==  0:
			return 0, 10
		list1 = []
		list2 = []
		for i in listOfGini:
			list1.append(i[0])
			list2.append(i[1])
			
		min = 10
		minIndex = 0
		for i in range(len(list1)):
			if list1[i] < min:
				min = list1[i]
				minIndex = list2[i]
		return minIndex, min
		
	def maxInfoGain_miss(self, dataFrame):
		listOfOutcomes = Library.getColumn(self, dataFrame, len(dataFrame.columns) - 1)
		probList = Library.getProbabilityList(self, listOfOutcomes)
		
		listOfEntropy = []
		# Loops through all the columns
		for i in range(1, len(dataFrame.columns) - 1):
			column = Library.getColumn(self, dataFrame, i)
			uniqueValues = self.getUniqueValues(column)
			uniqueKeys = self.getUniqueKeys(column)

			total = 0
			probMax = 0
			# Loops through all the unique values in the column (ex: A,G,T,C)
			for j in range(len(uniqueValues)):
				partialDataFrame = self.getPartialDataFrame(dataFrame, int(i), uniqueKeys[j])
				listOf_ithAttribute = Library.getColumn(self, partialDataFrame, len(partialDataFrame.columns) - 1)
				missClassificationError = self.calculateMiss(listOf_ithAttribute)
				prob = uniqueValues[j]/sum(uniqueValues)
				total = total + ( prob * missClassificationError )
			
			#appends the Information gain (entropy) to a list
			listOfEntropy.append([total, i])
		columnIndex, maximumIg = self.getMinError(listOfEntropy)
		
		return columnIndex, maximumIg, probList
	
	def buildTree_Chi_miss(self, dataFrame, prevChi):
		columnIndex, ig, probList = self.maxInfoGain_miss(dataFrame)
		tree = Tree()
		
		chi_df = self.getChiDataFrame(dataFrame, columnIndex)
		if columnIndex == 0:
			tree.data = self.getResult(dataFrame)
			tree.ig = 0
			tree.probList = probList
			tree.branches = []
			return tree
		
		tree.data = self.getColumnName(dataFrame, columnIndex)
		tree.ig = ig
		tree.probList = probList
		
		column = Library.getColumn(self, dataFrame, columnIndex)
		columnKeys = self.getUniqueKeys(column)
		
		treePrune = copy.deepcopy(tree)
		for i in columnKeys:
			columnDataFrame = self.getPartialDataFrame(dataFrame, columnIndex, i)
			columnDataFrame = columnDataFrame.drop(columnDataFrame.columns[[columnIndex]], axis = 1)
			treeTemp = copy.deepcopy(tree)
			treeTemp.data = self.getMostProbable(columnDataFrame)
			treePrune.addNode(i, treeTemp)
		
		chiSquareValue, dof = self.chiSquare(prevChi, chi_df)

		criticalValue = self.getCriticalValue(dof, self.confidence)
		if criticalValue < round(chiSquareValue, 2) or dof == 0:
			for i in columnKeys:
				columnDataFrame = self.getPartialDataFrame(dataFrame, columnIndex, i)
				columnDataFrame = columnDataFrame.drop(columnDataFrame.columns[[columnIndex]], axis = 1)
				dfp = copy.deepcopy(chi_df)
				tree.addNode(i, self.buildTree_Chi_miss(columnDataFrame, dfp))
		else:
			tree = copy.deepcopy(treePrune)
		return tree
	
	def buildTree_miss(self, dataFrame):
		columnIndex, ig, probList = self.maxInfoGain_miss(dataFrame)
		tree = Tree()
		
		if ig == 10:
			finalNode = Tree()
			tree.data = self.getResult(dataFrame)
			tree.ig = 0
			tree.probList = probList
			tree.branches = []
			return tree
		
		tree.data = self.getColumnName(dataFrame, columnIndex)
		tree.ig = ig
		tree.probList = probList
		
		column = Library.getColumn(self, dataFrame, columnIndex)
		columnKeys = self.getUniqueKeys(column)

		for i in columnKeys:
			columnDataFrame = self.getPartialDataFrame(dataFrame, columnIndex, i)
			columnDataFrame = columnDataFrame.drop(columnDataFrame.columns[[columnIndex]], axis = 1)
			tree.addNode(i, self.buildTree_miss(columnDataFrame))
		return tree
	
	def build(self, data, postPrune, chi, confidence):
		# Runs the required method build tree (Chi or without split stopping)
		if chi == True:
			self.confidence = confidence
			df_empty = pd.DataFrame({'P' : []})
			self.decisionTree = self.buildTree_Chi_miss(self.trainingData, df_empty)
			if postPrune == True:
				self.postPrune(self.decisionTree)
		else:
			self.decisionTree = self.buildTree_miss(self.trainingData)
	
		
class CART(Library):
	def __self__(self):
		self.trainingData = pd.DataFrame()
		self.decisionTree = Tree()
		self.counter = 0
		
	def getMinGini(self, listOfGini):
		# Returns minimum gini index from the given values
		check = 0
		for i in listOfGini:
			if i[0] != 0.0:
				check = 1
		if check ==  0:
			return 0, 10
		list1 = []
		list2 = []
		for i in listOfGini:
			list1.append(i[0])
			list2.append(i[1])
			
		min = 10
		minIndex = 0
		for i in range(len(list1)):
			if list1[i] < min:
				min = list1[i]
				minIndex = list2[i]
		return minIndex, min
	
	def giniIndex(self, list):
		# Calculates gini impurity for a given attribute
		totalElements = Library.findTotal(self, list)
		gI = 1
		gIList = []
		for i in list:
			probOfElement = (i/totalElements)
			gIList.append(probOfElement * probOfElement)
			
		for i in gIList:
			gI = gI - i
		return gI

	def calculateGini(self, listOfOutcomes):       
		values = Counter(listOfOutcomes).values()
		valueList = []
		for i in values:
			valueList.append(i)
		return self.giniIndex(valueList)
		

	def maxInfoGain_gini(self, dataFrame):
		# Returns the column with maximum information gain from a given data frame
		listOfOutcomes = Library.getColumn(self, dataFrame, len(dataFrame.columns) - 1)
		probList = Library.getProbabilityList(self, listOfOutcomes)
		
		listOfEntropy = []
		# Loops through all the columns
		for i in range(1, len(dataFrame.columns) - 1):
			column = Library.getColumn(self, dataFrame, i)
			uniqueValues = self.getUniqueValues(column)
			uniqueKeys = self.getUniqueKeys(column)
			
			totalGini = 0
			# Loops through all the unique values in the column (ex: A,G,T,C)
			for j in range(len(uniqueValues)):
				partialDataFrame = self.getPartialDataFrame(dataFrame, int(i), uniqueKeys[j])
				listOf_ithAttribute = Library.getColumn(self, partialDataFrame, len(partialDataFrame.columns) - 1)
				gini = self.calculateGini(listOf_ithAttribute)
				prob = uniqueValues[j]/sum(uniqueValues)
				totalGini = totalGini + ( prob * gini )
			
			#appends the Information gain (gini) to a list
			listOfEntropy.append([totalGini, i])
		columnIndex, maximumIg = self.getMinGini(listOfEntropy)
		
		return columnIndex, maximumIg, probList
			

	def buildTree_gini(self, dataFrame):
		columnIndex, ig, probList = self.maxInfoGain_gini(dataFrame)
		tree = Tree()

		if ig == 10:
			finalNode = Tree()
			tree.data = self.getResult(dataFrame)
			tree.ig = 0
			tree.probList = probList
			tree.branches = []
			return tree
		
		tree.data = self.getColumnName(dataFrame, columnIndex)
		tree.ig = ig
		tree.probList = probList
		
		column = Library.getColumn(self, dataFrame, columnIndex)
		columnKeys = self.getUniqueKeys(column)

		for i in columnKeys:
			columnDataFrame = self.getPartialDataFrame(dataFrame, columnIndex, i)
			columnDataFrame = columnDataFrame.drop(columnDataFrame.columns[[columnIndex]], axis = 1)
			tree.addNode(i, self.buildTree_gini(columnDataFrame))
		return tree
	

	def buildTree_Chi_gini(self, dataFrame, prevChi):
		columnIndex, ig, probList = self.maxInfoGain_gini(dataFrame)
		tree = Tree()
		
		chi_df = self.getChiDataFrame(dataFrame, columnIndex)
		if columnIndex == 0:
			tree.data = self.getResult(dataFrame)
			tree.ig = 0
			tree.probList = probList
			tree.branches = []
			return tree
		
		tree.data = self.getColumnName(dataFrame, columnIndex)
		tree.ig = ig
		tree.probList = probList
		
		column = Library.getColumn(self, dataFrame, columnIndex)
		columnKeys = self.getUniqueKeys(column)
		
		treePrune = copy.deepcopy(tree)
		for i in columnKeys:
			columnDataFrame = self.getPartialDataFrame(dataFrame, columnIndex, i)
			columnDataFrame = columnDataFrame.drop(columnDataFrame.columns[[columnIndex]], axis = 1)
			treeTemp = copy.deepcopy(tree)
			treeTemp.data = self.getMostProbable(columnDataFrame)
			treePrune.addNode(i, treeTemp)
		
		chiSquareValue, dof = self.chiSquare(prevChi, chi_df)
		
		criticalValue = self.getCriticalValue(dof, self.confidence)
		print(round(chiSquareValue, 2), criticalValue, dof)
		if criticalValue > round(chiSquareValue, 2) or dof == 0:
			print('accept')
			for i in columnKeys:
				columnDataFrame = self.getPartialDataFrame(dataFrame, columnIndex, i)
				columnDataFrame = columnDataFrame.drop(columnDataFrame.columns[[columnIndex]], axis = 1)
				dfp = copy.deepcopy(chi_df)
				tree.addNode(i, self.buildTree_Chi_gini(columnDataFrame, dfp))
		else:
			tree = copy.deepcopy(treePrune)
		return tree
	
	
	def build(self, data, postPrune, chi, confidence):
		# Runs the required method build tree (Chi or without split stopping)
		if chi == True:
			self.confidence = confidence
			df_empty = pd.DataFrame({'P' : []})
			self.decisionTree = self.buildTree_Chi_gini(self.trainingData, df_empty)
			if postPrune == True:
				self.postPrune(self.decisionTree)
		else:
			self.decisionTree = self.buildTree_gini(self.trainingData)
		

class ID3(Library):
	def __self__(self):
		self.trainingData = pd.DataFrame()
		self.decisionTree = Tree()
		self.counter = 0
		self.accuracyFinal = 0
		
	
	def Entropy(self, list):
		totalElements = Library.findTotal(self, list)
		entropy = 0
		for i in list:
			probOfElement = (i/totalElements)
			entropy = entropy + - ((probOfElement * np.log2(probOfElement)))

		return entropy

 
	def calculateEntropy(self, listOfOutcomes):       
		values = Counter(listOfOutcomes).values()
			
		valueList = []
		for i in values:
			valueList.append(i)
		calculationList = valueList
		entropy = self.Entropy(calculationList)
		return entropy
	

	def getMinEntropy(self, listOfEntropy):
		list1 = []
		list2 = []
		
		check = 0
		for i in listOfEntropy:
			if i[0] != 0.0:
				check = 1
		if check ==  0:
			return 0, 10
		
		for i in listOfEntropy:
			list1.append(i[0])
			list2.append(i[1])
		
		max = 0
		maxIndex = 0
		for i in range(len(list1)):
			if list1[i] > max:
				max = list1[i]
				maxIndex = list2[i]
		return maxIndex, max
	

	def buildTree_Chi(self, dataFrame, prevChi):
		
		columnIndex, ig, probList = self.maxInfoGain(dataFrame)
		tree = Tree()
		
		chi_df = self.getChiDataFrame(dataFrame, columnIndex)
		
		if columnIndex == 0:
			finalNode = Tree()
			tree.data = self.getResult(dataFrame)
			tree.ig = 0
			tree.probList = probList
			tree.branches = []
			return tree
		
		tree.data = self.getColumnName(dataFrame, columnIndex)
		tree.ig = ig
		tree.probList = probList
		
		column = Library.getColumn(self, dataFrame, columnIndex)
		columnKeys = self.getUniqueKeys(column)
		
		treePrune = copy.deepcopy(tree)
		for i in columnKeys:
			columnDataFrame = self.getPartialDataFrame(dataFrame, columnIndex, i)
			columnDataFrame = columnDataFrame.drop(columnDataFrame.columns[[columnIndex]], axis = 1)
			treeTemp = copy.deepcopy(tree)
			treeTemp.data = self.getMostProbable(columnDataFrame)
			treePrune.addNode(i, treeTemp)
		
		chiSquareValue, dof = self.chiSquare(prevChi, chi_df)

		criticalValue = self.getCriticalValue(dof, self.confidence)
		if criticalValue < round(chiSquareValue, 2) or dof == 0:
			for i in columnKeys:
				columnDataFrame = self.getPartialDataFrame(dataFrame, columnIndex, i)
				columnDataFrame = columnDataFrame.drop(columnDataFrame.columns[[columnIndex]], axis = 1)
				dfp = copy.deepcopy(chi_df)
				tree.addNode(i, self.buildTree_Chi(columnDataFrame, dfp))
		else:
			tree = copy.deepcopy(treePrune)
			
		return tree

	def buildTree(self, dataFrame):
		columnIndex, ig, probList = self.maxInfoGain(dataFrame)
		tree = Tree()
		
		if ig == 10:
			finalNode = Tree()
			tree.data = self.getResult(dataFrame)
			tree.ig = 0
			tree.probList = probList
			tree.branches = []
			return tree
		
		tree.data = self.getColumnName(dataFrame, columnIndex)
		tree.ig = ig
		tree.probList = probList

		column = Library.getColumn(self, dataFrame, columnIndex)
		columnKeys = self.getUniqueKeys(column)

		for i in columnKeys:
			columnDataFrame = self.getPartialDataFrame(dataFrame, columnIndex, i)
			columnDataFrame = columnDataFrame.drop(columnDataFrame.columns[[columnIndex]], axis = 1)
			tree.addNode(i, self.buildTree(columnDataFrame))
		
		return tree
	
	
	def maxInfoGain(self, dataFrame):
		# Returns the column with maximum information gain from a given data frame
		listOfOutcomes = Library.getColumn(self, dataFrame, len(dataFrame.columns) - 1)
		totalEntropy = self.calculateEntropy(listOfOutcomes)
		probList = self.getProbabilityList(listOfOutcomes)
		
		listOfEntropy = []
		# Loops through all the columns
		for i in range(1, len(dataFrame.columns) - 1):
			column = Library.getColumn(self, dataFrame, i)
			uniqueValues = self.getUniqueValues(column)
			uniqueKeys = self.getUniqueKeys(column)
			
			entropy = 0
			# Loops through all the unique values in the column (ex: A,G,T,C)
			for j in range(len(uniqueValues)):
				partialDataFrame = self.getPartialDataFrame(dataFrame, int(i), uniqueKeys[j])
				listOf_ithAttribute = Library.getColumn(self, partialDataFrame, len(partialDataFrame.columns) - 1)
				entropyOf_ithAttribute = self.calculateEntropy(listOf_ithAttribute)
				prob = uniqueValues[j]/sum(uniqueValues)
				entropy = entropy + ( prob * entropyOf_ithAttribute )
		
			#appends the Information gain (entropy) to a list
			listOfEntropy.append([totalEntropy - entropy, i])
		columnIndex, maximumIg = self.getMinEntropy(listOfEntropy)
		
		return columnIndex, maximumIg, probList
	
	def build(self, data, postPrune, chi, confidence):
		# Runs the required method build tree (Chi or without split stopping)
		if chi == True:
			self.confidence = confidence
			df_empty = pd.DataFrame({'P' : []})
			self.decisionTree = self.buildTree_Chi(self.trainingData, df_empty)
			if postPrune == True:
				self.postPrune(self.decisionTree)
		else:
			self.decisionTree = self.buildTree(self.trainingData)
	