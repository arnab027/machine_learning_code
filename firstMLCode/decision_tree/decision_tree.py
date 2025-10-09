from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from six import StringIO
# from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus
from sklearn.tree import export_graphviz

"""Decision Tree"""
"""git-url - https://github.com/Jack-Cherish/Machine-Learning/blob/master/Decision%20Tree/Sklearn-Decision%20Tree.py"""

if __name__ == '__main__':
	with open('decision_tree\lenses.txt', 'r') as fr:										#Load file
		lenses = [inst.strip().split('\t') for inst in fr.readlines()]		#process files
	lenses_target = []														#Extract the category of each set of data and save it in a list
	for each in lenses:
		lenses_target.append(each[-1])
	# print(lenses_target)

	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']			#feature label	
	lenses_list = []														#A temporary list to store lens data
	lenses_dict = {}														#A dictionary that stores lens data and is used to generate pandas
	for each_label in lensesLabels:											#Extract information and generate dictionary
		for each in lenses:
			lenses_list.append(each[lensesLabels.index(each_label)])
		lenses_dict[each_label] = lenses_list
		lenses_list = []
	# print(lenses_dict)														#Print dictionary information
	lenses_pd = pd.DataFrame(lenses_dict)									#Generate pandas.DataFrame
	# print(lenses_pd)														#Printing a pandas.DataFrame
	le = LabelEncoder()														#Create a LabelEncoder() object for serialization			
	for col in lenses_pd.columns:											#serialization
		lenses_pd[col] = le.fit_transform(lenses_pd[col])
	print(lenses_pd)														#Print encoding information

	clf = tree.DecisionTreeClassifier(max_depth = 4)						#Create the DecisionTreeClassifier() class
	clf = clf.fit(lenses_pd.values.tolist(), lenses_target)					#Using data to build a decision tree

	dot_data = StringIO()
	tree.export_graphviz(clf, out_file = dot_data,							#Draw decision tree
						feature_names = lenses_pd.keys(),
						class_names = clf.classes_,
						filled=True, rounded=True,
						special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf("D:\\pythonMachineLearning\\firstMLCode\\decision_tree\\tree.pdf") #Save the drawn decision tree in PDF format.
    
	print(clf.predict([[1,1,1,0]]))											#predict
