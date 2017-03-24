from numpy import *
import operator
import sys
sys.path.append('E:\MeachineLearn\AdaBoost\boost.py');

import boost

dataArr,labelArr = boost.loadDataSet('horseColicTraining2.txt');
weakClassArr = boost.adaBoost(dataArr,labelArr)
print(weakClassArr)
testArr,testlabelArr = boost.loadDataSet('horseColicTest2.txt');
prediction = boost.adaClassify(testArr,weakClassArr)
error = mat(ones((67,1)))
errorRate = error[prediction!=mat(testlabelArr).T].sum()/float(67.0)
print(errorRate)

