from numpy import *

def loadDataSet():
    dataArr = matrix([
        [1.0,2.1],
        [2,1.1],
        [1.3,1],
        [1.0,1.0],
        [2.0,1.0]
    ]);
    classArr = [1.0,1.0,-1.0,-1.0,1.0];
    return dataArr,classArr

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = [];labelMat = [];
    fr = open(filename)
    fileLines = fr.readlines()
    for line in fileLines:
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMat,dimen,threshVal,threshIneq):
    m,n = shape(dataMat)
    arr = ones((m,1))
    if threshIneq == 'lt':
        arr[dataMat[:,dimen] <= threshVal] = -1.0
    else:
        arr[dataMat[:,dimen] > threshVal] = -1.0
    return arr

def buildStump(dataArr,classArr,D):
    dataMatrix = mat(dataArr);classLabel = mat(classArr).transpose()
    m,n = shape(dataMatrix)
    numSteps = 10
    minerror = inf
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    for i in range(n):
        colmax = dataMatrix[:,i].max();colmin = dataMatrix[:,].min()
        stepSize = (colmax - colmin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for threshIneq in ('lt','gt'):
                threshVal = colmin + stepSize*j
                preditctedArr = stumpClassify(dataMatrix,i,threshVal,threshIneq)
                errArr = mat(ones((m,1)))
                errArr[preditctedArr==classLabel]=0
                weightedError = D.T*errArr
               # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, threshIneq, weightedError) )
                if weightedError < minerror:
                    minerror = weightedError
                    bestClassEst = preditctedArr.copy()
                    bestStump['dim'] = i
                    bestStump['ineq'] = threshIneq
                    bestStump['threshVal'] = threshVal
    return bestStump,minerror,bestClassEst

def adaBoost(dataArr,classArr,numInt = 30):
    m,n = shape(dataArr)
    D = mat(ones((m,1))/m)
    classMat = mat(classArr)
    aggClassEst = mat(zeros((m,1)));
    weakClassArr = []
    for i in range(numInt):
        stump,errors,classEst = buildStump(dataArr,classArr,D);
        alpha = float( 0.5*log((1-errors)/max(errors,1e-16)) )
        stump['alpha'] = alpha
        weakClassArr.append(stump)
        expon = multiply(-alpha*classMat.T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst+=classEst*alpha;
        aggClassErrors = multiply(sign(aggClassEst)!=classMat.T,mat(ones((m,1))))
        if aggClassErrors.sum() == 0.0: break
    return weakClassArr;

def adaClassify(data,weakClassArr):
    datamat = mat(data)
    m = shape(datamat)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(weakClassArr)):
        classEst = stumpClassify(datamat,weakClassArr[i]['dim'],weakClassArr[i]['threshVal'],weakClassArr[i]['ineq']) #写错了
        aggClassEst += weakClassArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)