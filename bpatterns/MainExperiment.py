import numpy as np
import pandas as pd
import math

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn import metrics, datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


from commodm.bpatterns.SplitData import splitData
from commodm.bpatterns.GetColumnSubtable import buidDecisionColumnDataSets

def myRound100(fvalue):
    value = round(fvalue, 2)
    return value

def myRound1000(fvalue):
    value = round(fvalue, 3)
    return value

def getModelByName(modelName):

    if modelName=="KNeighborsClassifier":
        return KNeighborsClassifier(n_neighbors=3, metric='euclidean')

    if modelName=="DecisionTreeClassifier":
        return DecisionTreeClassifier()


    if modelName=="GaussianNB":
        return GaussianNB()

    if modelName == "MLPClassifier":
        return MLPClassifier()

    if modelName == "LogisticRegression":
        return LogisticRegression()

    if modelName == "RandomForestClassifier":
        return RandomForestClassifier()

    if modelName == "AdaBoostClassifier":
        return AdaBoostClassifier(algorithm="SAMME")

    if modelName == "GradientBoostingClassifier":
        return GradientBoostingClassifier()

    if modelName == "XGBClassifier":
        return XGBClassifier()

    print("UNKNOWN CLASSIFIER MODEL:",modelName)
    exit()

#===========================================================

def selectAttr(features,labels,expectedFeaturesNumber):


    model = RandomForestClassifier()

    model.fit(features, np.ravel(labels))  # Uczenie klasyfikatora


    wagi_cech = model.feature_importances_
    columns = list(features.columns.values)

    columnsWeights = []
    for i in range(0, len(columns)):
        pair = (columns[i], float(wagi_cech[i]), i)
        columnsWeights.append(pair)

    def objCompareFunc(element):
        return element[1]

    columnsWeights.sort(reverse=True, key=objCompareFunc)

    selected_attr_index_list = []

    for i in range(0, len(columnsWeights)):
        para = columnsWeights[i]

        if i>expectedFeaturesNumber-1: break

        selected_attr_index_list.append(para[2])


    return selected_attr_index_list

def selectAttrBySelectKBestMutualInfo(features,labels,noSelectedFeatures):

    dlabels = np.ravel(labels)

    model = SelectKBest(score_func=mutual_info_classif, k=noSelectedFeatures)

    model = model.fit(features, np.ravel(dlabels))

    selected_info = model.get_support()


    selected_attr_index_list = []
    for i in range(0, len(selected_info)):
        if selected_info[i] == True:
            selected_attr_index_list.append(i)


    print(selected_attr_index_list)

    return selected_attr_index_list

def selectAttrBySelectKBestFClassif(features,labels,noSelectedFeatures):

    dlabels = np.ravel(labels)

    model = SelectKBest(score_func=f_classif, k=noSelectedFeatures)

    model = model.fit(features, np.ravel(dlabels))

    selected_info = model.get_support()

    selected_attr_index_list = []
    for i in range(0, len(selected_info)):
        if selected_info[i] == True:
            selected_attr_index_list.append(i)

    return selected_attr_index_list



#===========================================================

def changeDataIdentification(dataset,clusters,takeDECISION_NOW):
    noRow = dataset.shape[0]
    kolumny = list(dataset.columns.values)

    if takeDECISION_NOW:
        firstDecisionAttrIndex = 1905
    else:
        firstDecisionAttrIndex = 1899

    column_list = []
    for i in range(0, len(kolumny)):
        if (i > 0 and i < firstDecisionAttrIndex) or i == 1907:
            column_list.append(i)

    sel_dataset = dataset.iloc[:, column_list]

    noCol = sel_dataset.shape[1]

    for i in range(0, noRow):
        new_dec = clusters[i]
        sel_dataset.iat[i, noCol - 1] = new_dec

    return sel_dataset

def changeDataPrediction(dataset,clusters, takeDECISION_NOW, selectedIndexList):
    noRow = dataset.shape[0]
    kolumny = list(dataset.columns.values)

    if takeDECISION_NOW:
        firstDecisionAttrIndex = 1905
    else:
        firstDecisionAttrIndex = 1899

    column_list = []
    for i in range(0, len(kolumny)):
        if (i > 0 and i < firstDecisionAttrIndex) or i == 1907:
            column_list.append(i)

    sel_dataset = dataset.iloc[:, column_list]

    noCol = sel_dataset.shape[1]

    for i in range(0, noRow):
        new_dec = clusters[i]
        sel_dataset.iat[i, noCol - 1] = new_dec

    sel_obj_dataset = sel_dataset.iloc[selectedIndexList, :]

    return sel_obj_dataset



def getBPattern(mean,min,max):
    if mean < 0.3 and min >= 0.0 and max < 0.4:
        return 0
    else:
        if mean > 0.9 and min > 0.8 and max <= 1.0:
            return 2
        else:
            return 1


def predictToBPattern(identOnly,dataset,testAllUpDownPrediction):

    # 0 - all
    # 1 - worse
    # 2 - better
    # 3 - stable

    bpatters = []
    selectedIndexListTest = []

    noRow = dataset.shape[0]
    for i in range(0, noRow):
        meanNext = float(dataset.iat[i, 0])
        minNext = float(dataset.iat[i, 1])
        maxNext = float(dataset.iat[i, 2])

        nextPattern = getBPattern(meanNext,minNext,maxNext)

        meanNow = float(dataset.iat[i, 3])
        minNow = float(dataset.iat[i, 4])
        maxNow = float(dataset.iat[i, 5])

        nowPattern = getBPattern(meanNow, minNow, maxNow)

        if identOnly:
            bpatters.append(nowPattern)
        else:
            bpatters.append(nextPattern)

        if testAllUpDownPrediction==0:
            selectedIndexListTest.append(i)
        else:
            if testAllUpDownPrediction == 1:
                if nextPattern<nowPattern:
                    selectedIndexListTest.append(i)
            else:
                if testAllUpDownPrediction == 2:
                    if nextPattern > nowPattern:
                        selectedIndexListTest.append(i)
                else:
                    if testAllUpDownPrediction == 3:  # Testujemy przejscia bez zmian
                        if nextPattern == nowPattern:
                            selectedIndexListTest.append(i)
                    else:
                        print("Unexpected value testAllUpDownPrediction:",testAllUpDownPrediction)
                        exit()



    return bpatters, selectedIndexListTest

def dataPreparation(SHOW_DETAILS,identOnly,takeDECISION_NOW,
                    train_data,test_data, decision_columns_train_data, decision_columns_test_data,
                    testAllUpDownPrediction):

    if SHOW_DETAILS: print("Constructing decision classes based on temporal patterns...")

    # Ustawiamy testAllUpDownPredition=0 bo to tablica treningowa i zawsze twestujemy wszystkie obiekty
    clusters_train, selectedIndexListTrain = predictToBPattern(identOnly, decision_columns_train_data,0)

    clusters_test, selectedIndexListTest = predictToBPattern(identOnly, decision_columns_test_data,testAllUpDownPrediction)

    if SHOW_DETAILS: print("Changing decision...")


    if identOnly:
        dec_train_data = changeDataIdentification(train_data, clusters_train,takeDECISION_NOW)
        dec_test_data = changeDataIdentification(test_data, clusters_test,takeDECISION_NOW)
    else:
        dec_train_data = changeDataPrediction(train_data, clusters_train, takeDECISION_NOW, selectedIndexListTrain)
        dec_test_data = changeDataPrediction(test_data, clusters_test, takeDECISION_NOW, selectedIndexListTest)

    return dec_train_data, dec_test_data


def makeExperiment(SHOW_DETAILS,filtrName,identOnly, takeDECISION_NOW, expectedFeaturesNumber,CLASSIFIER_NAME,
                   testAllUpDownPrediction,USE_THRESHOLDS,threshold0,threshold1):

    data = pd.read_csv("./data/bpdata2.txt", sep=";") #Odczytanie głównej tablicy z danymi

    if SHOW_DETAILS: print("RANDOM DATA SPLIT (ID SPLITTING)...")
    train_table,test_table = splitData(data)

    decision_columns_train_data, decision_columns_test_data = buidDecisionColumnDataSets(train_table,test_table)

    if SHOW_DETAILS: print("BUILDING DATA FOR THE EXPERIMENT...")
    traindataset, testdataset = dataPreparation(SHOW_DETAILS,identOnly,takeDECISION_NOW,
                                                train_table,test_table,
                                                decision_columns_train_data, decision_columns_test_data,
                                                testAllUpDownPrediction)
    #--------------------------------------------------

    noColumn = traindataset.shape[1]

    features_train = traindataset.iloc[:,1:noColumn-1]

    labels_train_float = traindataset.iloc[:,[noColumn-1]]
    labels_train_float = np.ravel(labels_train_float);

    labels_train = []
    for i in range(0, len(labels_train_float)):
        dec = int(labels_train_float[i])
        labels_train.append(dec)


    #Features selection
    if filtrName=="FILTR1":
        selected_attr_index_list = selectAttr(features_train,labels_train,expectedFeaturesNumber)
    else:
        if filtrName == "FILTR2":
            selected_attr_index_list = selectAttrBySelectKBestFClassif(features_train, labels_train, expectedFeaturesNumber)
        else:
            if filtrName == "FILTR3":
                selected_attr_index_list = selectAttrBySelectKBestMutualInfo(features_train, labels_train, expectedFeaturesNumber)
            else:
                print("Unknown filtr ",filtrName)
                exit()



    features_train = features_train.iloc[:,selected_attr_index_list]


    features_test = testdataset.iloc[:,1:noColumn-1]
    features_test = features_test.iloc[:,selected_attr_index_list]

    labels_test_float = testdataset.iloc[:,[noColumn-1]]
    labels_test_float = np.ravel(labels_test_float);
    labels_test = []
    for i in range(0,len(labels_test_float)):
        dec = int(labels_test_float[i])
        labels_test.append(dec)


    model = getModelByName(CLASSIFIER_NAME)

    model.fit(features_train, labels_train)

    labels_predicted_prob = model.predict_proba(features_test)

    classes = model.classes_

    selectedDecValIndex0 = -1
    for i in range(0, len(classes)):
        if classes[i] == 0:
            selectedDecValIndex0 = i
            break

    if selectedDecValIndex0 == -1:
        print("No values 0")
        exit()

    #------------------------------

    selectedDecValIndex1 = -1
    for i in range(0, len(classes)):
        if classes[i] == 1:
            selectedDecValIndex1 = i
            break

    if selectedDecValIndex1 == -1:
        print("No values 1")
        exit()

    #------------------------------

    selectedDecValIndex2 = -1
    for i in range(0, len(classes)):
        if classes[i] == 2:
            selectedDecValIndex2 = i
            break

    if selectedDecValIndex2 == -1:
        print("No values 2")
        exit()

    #------------------------------


    labels_predicted = []


    for i in range(0, len(labels_predicted_prob)):

        gen_prob0 = labels_predicted_prob[i][selectedDecValIndex0]
        gen_prob1 = labels_predicted_prob[i][selectedDecValIndex1]
        gen_prob2 = labels_predicted_prob[i][selectedDecValIndex2]


        if USE_THRESHOLDS:

            if gen_prob0 > threshold0:
                labels_predicted.append(0)
            else:
                if gen_prob1 > threshold1:
                    labels_predicted.append(1)
                else:
                    labels_predicted.append(2)
        else:

            if gen_prob0>gen_prob1 and gen_prob0>gen_prob2:
                labels_predicted.append(0)
            else:
                if gen_prob1 > gen_prob0 and gen_prob1 > gen_prob2:
                    labels_predicted.append(1)
                else:
                    labels_predicted.append(2)

    # Calculating classification quality by comparing: labels_predicted and labels_test

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)

    if SHOW_DETAILS: print("Accuracy=", accuracy)

    if SHOW_DETAILS: print("========= FOOL RESULTS ================")

    report = classification_report(labels_test, labels_predicted,  output_dict=True, zero_division=0)


    try:
        recall0 = report['0']['recall']
    except KeyError as myerror:  # Łapanie wyjątku
        recall0 = 0.0

    recall1 = report['1']['recall']

    try:
        recall2 = report['2']['recall']
    except KeyError as myerror:  # Łapanie wyjątku
        recall2 = 0.0

    if SHOW_DETAILS:
        print(report)

        print("====== Confusion matrix =========")

        conf_matrix = confusion_matrix(labels_test, labels_predicted)
        print(conf_matrix)

    f1score = f1_score(labels_test, labels_predicted, average='macro')


    return  f1score, accuracy, recall0, recall1, recall2



def myformat(number):
    return "{:>5.3f}".format(number)

def runOneEXPERIMENT(filtrName,IDENTIFICATION_ONLY,DECISION_NOW_AS_CONDITION,FEATURE_NUMBER,CLASSIFIER_NAME,
                     testAllUpDownPrediction, USE_THRESHOLD, threshold0, threshold1):

    SHOW_DETAILS = False

    f1scoreList = []
    accuracyList = []
    recall0_List = []
    recall1_List = []
    recall2_List = []
    K=10 #Number of experiments

    sum_f1score = 0.0
    sum_accuracy = 0.0
    sum_recall0 = 0.0
    sum_recall1 = 0.0
    sum_recall2 = 0.0

    for k in range(0,K):
        f1score, accuracy, recall0, recall1, recall2 = (
            makeExperiment(SHOW_DETAILS,filtrName,IDENTIFICATION_ONLY,DECISION_NOW_AS_CONDITION,FEATURE_NUMBER,CLASSIFIER_NAME,
                           testAllUpDownPrediction,USE_THRESHOLD,threshold0,threshold1))

        f1scoreList.append(f1score)
        accuracyList.append(accuracy)
        recall0_List.append(recall0)
        recall1_List.append(recall1)
        recall2_List.append(recall2)

        sum_f1score = sum_f1score + f1score
        sum_accuracy = sum_accuracy + accuracy
        sum_recall0 = sum_recall0 + recall0
        sum_recall1 = sum_recall1 + recall1
        sum_recall2 = sum_recall2 + recall2

        #print(accuracy, recall0, recall1, recall2)

    mean_f1score = sum_f1score / K
    mean_accuracy = sum_accuracy / K
    mean_recall0 = sum_recall0 / K
    mean_recall1 = sum_recall1 / K
    mean_recall2 = sum_recall2 / K

    stddev_f1score = 0.0
    stddev_accuracy = 0.0
    stddev_recall0 = 0.0
    stddev_recall1 = 0.0
    stddev_recall2 = 0.0

    for k in range(0, K):
        stddev_f1score = stddev_f1score + (f1scoreList[k] - mean_f1score) * (f1scoreList[k] - mean_f1score)
        stddev_accuracy = stddev_accuracy + (accuracyList[k]-mean_accuracy)*(accuracyList[k]-mean_accuracy)
        stddev_recall0 = stddev_recall0 + (recall0_List[k] - mean_recall0) * (recall0_List[k] - mean_recall0)
        stddev_recall1 = stddev_recall1 + (recall1_List[k] - mean_recall1) * (recall1_List[k] - mean_recall1)
        stddev_recall2 = stddev_recall2 + (recall2_List[k] - mean_recall2) * (recall2_List[k] - mean_recall2)

    stddev_f1score = math.sqrt(stddev_f1score / (K - 1))
    stddev_accuracy = math.sqrt(stddev_accuracy/(K-1))
    stddev_recall0 = math.sqrt(stddev_recall0 / (K - 1))
    stddev_recall1 = math.sqrt(stddev_recall1 / (K - 1))
    stddev_recall2 = math.sqrt(stddev_recall2 / (K - 1))

    tekst = str(filtrName)+ "|"+ str(IDENTIFICATION_ONLY) + "|" + str(DECISION_NOW_AS_CONDITION) + "|" + str(FEATURE_NUMBER) + "|" + str(CLASSIFIER_NAME) + "|"
    tekst = tekst + myformat(mean_f1score) + "|" + myformat(stddev_f1score) + "|"
    tekst = tekst + myformat(mean_accuracy) + "|" + myformat(stddev_accuracy) + "|"
    tekst = tekst + myformat(mean_recall0) + "|" + myformat(stddev_recall0) + "|"
    tekst = tekst + myformat(mean_recall1) + "|" + myformat(stddev_recall1) + "|"
    tekst = tekst + myformat(mean_recall2) + "|" + myformat(stddev_recall2) + "|"

    return tekst

#===========================================================================================




def testDifferentTransactions():

    print("*** TEST DIFFERENT TRANSACTIONS ***")

    IDENTIFICATION_ONLY = False
    takeDECISION_NOW_AS_CONDITION = False
    FEATURE_NUMBER = 1000
    CLASSIFIER_NAME = "XGBClassifier"

    USE_THRESHOLD = True
    threshold0 = 0.2
    threshold1 = 0.2

    resultList = []

    print("All")

    testAllUpDownPrediction = 0
    results = runOneEXPERIMENT("FILTR1",IDENTIFICATION_ONLY, takeDECISION_NOW_AS_CONDITION, FEATURE_NUMBER, CLASSIFIER_NAME,
                              testAllUpDownPrediction,USE_THRESHOLD,threshold0,threshold1)
    myresult = "All|" + results
    resultList.append(myresult)
    print(myresult)

    print("Worse")

    testAllUpDownPrediction = 1
    results = runOneEXPERIMENT("FILTR1",IDENTIFICATION_ONLY, takeDECISION_NOW_AS_CONDITION, FEATURE_NUMBER, CLASSIFIER_NAME,
                               testAllUpDownPrediction,USE_THRESHOLD,threshold0,threshold1)
    myresult = "Worse|" + results
    resultList.append(myresult)
    print(myresult)

    print("Better")

    testAllUpDownPrediction = 2
    results = runOneEXPERIMENT("FILTR1",IDENTIFICATION_ONLY, takeDECISION_NOW_AS_CONDITION, FEATURE_NUMBER, CLASSIFIER_NAME,
                               testAllUpDownPrediction,USE_THRESHOLD,threshold0,threshold1)
    myresult = "Better|" + results
    resultList.append(myresult)
    print(myresult)

    print("Stable")

    testAllUpDownPrediction = 3
    results = runOneEXPERIMENT("FILTR1",IDENTIFICATION_ONLY, takeDECISION_NOW_AS_CONDITION, FEATURE_NUMBER, CLASSIFIER_NAME,
                               testAllUpDownPrediction,USE_THRESHOLD,threshold0,threshold1)
    myresult = "Stable|" + results
    resultList.append(myresult)
    print(myresult)

    print("--- RESULTS ---")

    for i in range(0,len(resultList)):
        print(resultList[i])

def testDifferentThresholds():

    print("*** TEST DIFFERENT THRESHOLDS ***")

    IDENTIFICATION_ONLY = False
    takeDECISION_NOW_AS_CONDITION = False
    FEATURE_NUMBER = 1000
    CLASSIFIER_NAME = "XGBClassifier"

    resultList = []

    testAllUpDownPrediction = 0

    USE_THRESHOLD = True

    t0 = float(0.1)
    while t0<1.0:
        print("  t0=", t0)
        t1 = float(0.1)
        while t1 < 1.0:
            print("     t1=", t1)

            results = runOneEXPERIMENT("FILTR1", IDENTIFICATION_ONLY, takeDECISION_NOW_AS_CONDITION, FEATURE_NUMBER, CLASSIFIER_NAME,
                                      testAllUpDownPrediction, USE_THRESHOLD, t0, t1)
            myresult = str(myRound100(t0)) +"|" + str(myRound100(t1)) + "|" + results
            resultList.append(myresult)

            t1 = t1 + 0.1

        t0 = t0 + 0.1


    print("--- RESULTS ---")

    for i in range(0,len(resultList)):
        print(resultList[i])

def testDifferentNumberOfFeatures(filtrName):

    print("*** TEST FOR DIFFERENT NUMBER OF FEATURES ***")

    IDENTIFICATION_ONLY = False
    takeDECISION_NOW_AS_CONDITION = False
    CLASSIFIER_NAME = "XGBClassifier"

    testAllUpDownPrediction = 0

    numberOfFeatureList =  [5,10,20,50,100,200,500,1000,1500]

    USE_THRESHOLD = False
    t0 = 0.2
    t1 = 0.2
    resultList = []

    for i in range(0,len(numberOfFeatureList)):
        n = numberOfFeatureList[i]

        print("Test for number of features=",n)

        results = runOneEXPERIMENT(filtrName,IDENTIFICATION_ONLY, takeDECISION_NOW_AS_CONDITION, n, CLASSIFIER_NAME,
                                   testAllUpDownPrediction, USE_THRESHOLD, t0, t1)
        myresult = str(n) + "|" + results
        #print("myresult=", myresult)
        resultList.append(myresult)

    print("--- RESULTS ---")

    for i in range(0,len(resultList)):
        print(resultList[i])


def testDifferentClassifiers():
    IDENTIFICATION_ONLY = False
    takeDECISION_NOW_AS_CONDITION = False

    testAllUpDownPrediction = 0

    CLASSFIERS = ["GaussianNB","KNeighborsClassifier","DecisionTreeClassifier","MLPClassifier","LogisticRegression","RandomForestClassifier","AdaBoostClassifier","GradientBoostingClassifier","XGBClassifier"]

    resultList = []

    print("*** TEST CLASSIFIERS ***")

    USE_THRESHOLD = False
    t0 = 0.2
    t1 = 0.2
    NO_FEATURES = 1000
    for i in range(0, len(CLASSFIERS)):
        CLASSIFIER_NAME = CLASSFIERS[i]

        print(" Test for:",CLASSIFIER_NAME)

        results = runOneEXPERIMENT("FILTR1", IDENTIFICATION_ONLY, takeDECISION_NOW_AS_CONDITION, NO_FEATURES, CLASSIFIER_NAME,
                                   testAllUpDownPrediction, USE_THRESHOLD, t0, t1)


        myresult = str(CLASSIFIER_NAME) + "|" + results

        resultList.append(myresult)

    print("--- RESULTS ---")

    for i in range(0, len(resultList)):
        print(resultList[i])



if __name__ == "__main__":

    testDifferentClassifiers()
    print("----------------------------------------------")
    testDifferentNumberOfFeatures("FILTR1") #"FILTR2 or "FILTR3""
    print("----------------------------------------------")
    testDifferentThresholds()
    print("----------------------------------------------")
    testDifferentTransactions()
    print("----------------------------------------------")

    print("STOP")
