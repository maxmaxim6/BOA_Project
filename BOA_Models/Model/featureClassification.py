from cProfile import label
from turtle import left
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import _joblib
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import csv
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest

from sklearn.metrics import average_precision_score
from  matplotlib.pyplot import figure, xlabel
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier, BaggingClassifier

from CategoryClassifier import CategoryClassifier
from xgboost import XGBClassifier

from translate_labels import transform_to_browser_labels,transform_to_browser_app_labels,transform_to_app_labels,transform_to_os_labels
import copy

def read_data_set(path, rows_to_skip):
    ds = pd.read_csv(path,skiprows=rows_to_skip,header=None)
    return ds

def trainKNN(X_train,y_train):
    #normalization
    scaler = preprocessing.StandardScaler().fit(X_train)
    filename = 'KNNTrainScalar.joblib.pkl'
    _joblib.dump(scaler, filename, compress=9)
    XTrainScaled = scaler.transform(X_train)

    # cross fitting
    neighbors_range = [2,4,8,16,32,64]
    #neighbors_range = [16,32]
    
    distance_types = ['chebyshev', 'sokalmichener',
    'canberra', #'haversine',
    #'rogerstanimoto', 'matching',
    'dice', 'euclidean',
    'braycurtis', 'russellrao',
    'cityblock', 'manhattan',
    #'infinity', 'jaccard',
    #'sokalsneath', # 'seuclidean',
    #'kulsinski', 'minkowski',
    #'mahalanobis', 'p',
    #'l2', 'hamming',
    #'l1', #'wminkowski',
    #'pyfunc']
    ]


    algorithms=['ball_tree']
    #y_train = transform_to_app_labels(copy.deepcopy(y_train))
    #y_train = transform_to_os_labels(copy.deepcopy(y_train))
    #y_train = transform_to_browser_labels(copy.deepcopy(y_train))
    print("y_train: ",y_train)
    param_grid = dict(algorithm=algorithms, n_neighbors=neighbors_range, metric=distance_types)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.30, random_state=42)
    cv.get_n_splits(X_train,y_train)
    #, n_jobs=-1, cv=cv, verbose=100
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid,n_jobs=-1,cv=cv, verbose=100)
    #print("y:!! ",y_train)
    grid.fit(XTrainScaled, y_train)

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    # train by best params
    n_neighbors = grid.best_params_['n_neighbors']
    metric = grid.best_params_['metric']

    clf_opt = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    
    clf_opt.fit(XTrainScaled, y_train)

    filename = 'KNN.joblib.pkl'
    _joblib.dump(clf_opt, filename, compress=9)

def testKNN(X_test,y_test):
    name = "KNN"
    clf = _joblib.load('KNN.joblib.pkl')
    scaler = _joblib.load('KNNTrainScalar.joblib.pkl')
    #y_test = transform_to_os_labels(copy.deepcopy(y_test))
    #y_test = transform_to_app_labels(copy.deepcopy(y_test))
    #y_test = transform_to_browser_labels(copy.deepcopy(y_test))
    print("y_test: ",y_test)
    X_testScaled = scaler.transform(X_test)
    y_pred = clf.predict(X_testScaled)
    print('KNN precision: ',metrics.precision_score(y_test, y_pred,average="micro"))
    print(' KNN accuracy: ',metrics.accuracy_score(y_test, y_pred))

    x = []
    y = []
    class_report  = classification_report(y_test, y_pred)
    out_name= 'accuracy_report_{}_.txt'.format(name)#,target_names=target_names
    with open(out_name, "w") as text_file:
        text_file.write(class_report)
        text_file.write('percision: %s'%metrics.precision_score(y_test, y_pred,average="micro"))
        text_file.write(' accuracy: %s'%metrics.accuracy_score(y_test, y_pred))
    print(class_report)
    labels = ['among_us', 'facebook', 'instagram', 'pubg','telegram','tien_len']
    columns = ['%s' %(i) for i in labels[0:len(np.unique(y_test))]]
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    print("cm: ",cm)
    df_cm = pd.DataFrame(cm, index=columns, columns=columns)
    
    sns.heatmap(df_cm,cmap='Blues' ,fmt=".0f", annot=True)
    plt.title("KNN confusion matrix")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    image_name = name + ".pdf"
    plt.savefig('confusion_matrix_{}.png'.format(name))

   
def trainRandomForest(X_train,y_train):
    #normalization
    scaler = preprocessing.StandardScaler().fit(X_train)
    filename = 'RandomForestTrainScalar.joblib.pkl'
    _joblib.dump(scaler, filename, compress=9)
    XTrainScaled = scaler.transform(X_train)
    
    # cross fitting
    n_estimators = [10,20,40,60,80,100]
    max_depth = [5,10]
    #max_features = [1,5,10,15]
    #, max_features = max_features
    param_grid = dict(n_estimators = n_estimators, max_depth = max_depth)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.30, random_state=42)
    cv.get_n_splits(X_train,y_train)
    grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv)
    print("XTrainScaled ",XTrainScaled)
    print("y_train ",y_train)
    grid.fit(XTrainScaled, y_train)

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    best_max_depth = grid.best_params_['max_depth']
    best_n_estimators = grid.best_params_['n_estimators']
    #best_max_features = grid.best_params_['max_features']
    #, max_features = best_max_features
    clf_opt =  RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth)
    clf_opt.fit(XTrainScaled, y_train)
    #MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
    filename = 'RandomForest.joblib.pkl'
    _joblib.dump(clf_opt, filename, compress=9)

def testRandomForest(X_test,y_test):
    name = "RandomForest"
    clf = _joblib.load('RandomForest.joblib.pkl')
    scaler = _joblib.load('RandomForestTrainScalar.joblib.pkl')
    X_testScaled = scaler.transform(X_test)
    y_pred = clf.predict(X_testScaled)
    print('RandomForest percision: ',metrics.precision_score(y_test, y_pred,average="micro"))
    print('RandomForest accuracy: ',metrics.accuracy_score(y_test, y_pred))
    
    class_report  = classification_report(y_test, y_pred)
    out_name= 'accuracy_report_{}_.txt'.format(name)#,target_names=target_names
    with open(out_name, "w") as text_file:
        text_file.write(class_report)
        text_file.write('percision: %s'%metrics.precision_score(y_test, y_pred,average="micro"))
        text_file.write(' accuracy: %s'%metrics.accuracy_score(y_test, y_pred))
    print(class_report)
    labels = ['among_us', 'facebook', 'instagram', 'pubg','telegram','tien_len']
    columns = ['%s' %(i) for i in labels[0:len(np.unique(y_test))]]
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    print("cm: ",cm)
    df_cm = pd.DataFrame(cm, index=columns, columns=columns)
    
    sns.heatmap(df_cm,cmap='Blues' ,fmt=".0f", annot=True)
    
    plt.title("Random Forest Normalized confusion matrix")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    image_name = name + ".pdf"
    
    plt.savefig('confusion_matrix_{}.png'.format(name))

    '''
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    eunique_label = np.unique(y_train)#.tolist()
    plot_confusion_matrix(cm_normalized, eunique_label,title='Normalized confusion matrix')
    image_name = name + ".pdf"
    plt.savefig(image_name)
    #precision, recall, threshold = precision_recall_curve(y_test, y_pred)
    '''

def trainSVC_RBF(X_train,y_train):
    #normalization
    scaler = preprocessing.StandardScaler().fit(X_train)
    filename = 'SVMRBFTrainScalar.joblib.pkl'
    _joblib.dump(scaler, filename, compress=9)
    XTrainScaled = scaler.transform(X_train)
    
    # cross fitting
    C_range = [2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(5),2**(5),2**(7),2**(9),2**(11),2**(13),2**(15)]
    gamma_range = [2**(-15),2**(-13),2**(-11),2**(-9),2**(-7),2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(3)]
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.30, random_state=42)
    cv.get_n_splits(X_train,y_train)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    print("XTrainScaled ",XTrainScaled)
    print("y_train ",y_train)
    grid.fit(XTrainScaled, y_train)

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    C=grid.best_params_['C']
    clf_ChromeTrainRBF = rbf_svc = SVC(kernel='rbf', gamma=grid.best_params_['gamma'], C=C)
    clf_ChromeTrainRBF.fit(XTrainScaled, y_train)
    #MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
    filename = 'SVMRBF.joblib.pkl'
    _joblib.dump(clf_ChromeTrainRBF, filename, compress=9)

def testSVC_RBF(X_test,y_test):

    name = "SVMRBF"
    clf = _joblib.load('SVMRBF.joblib.pkl')
    scaler = _joblib.load('SVMRBFTrainScalar.joblib.pkl')
    X_testScaled = scaler.transform(X_test)
    y_pred = clf.predict(X_testScaled)
    print('SVC percision: ',metrics.precision_score(y_test, y_pred,average="micro"))
    print('svc accuracy: ',metrics.accuracy_score(y_test, y_pred))

    class_report  = classification_report(y_test, y_pred)
    out_name= 'accuracy_report_{}_.txt'.format(name)#,target_names=target_names
    with open(out_name, "w") as text_file:
        text_file.write(class_report)
        text_file.write('percision: %s'%metrics.precision_score(y_test, y_pred,average="micro"))
        text_file.write(' accuracy: %s'%metrics.accuracy_score(y_test, y_pred))
    print(class_report)
    labels = ['among_us', 'facebook', 'instagram', 'pubg','telegram','tien_len']
    columns = ['%s' %(i) for i in labels[0:len(np.unique(y_test))]]
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    print("cm: ",cm)
    df_cm = pd.DataFrame(cm, index=columns, columns=columns)
    
    sns.heatmap(df_cm,cmap='Blues' ,fmt=".0f", annot=True)
    plt.title("SVM RBF Normalized confusion matrix")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    image_name = name + ".pdf"
    plt.savefig('confusion_matrix_{}.png'.format(name))

    '''
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    eunique_label = np.unique(y_train)#.tolist()
    plot_confusion_matrix(cm_normalized, eunique_label,title='Normalized confusion matrix')
    image_name = name + ".pdf"
    plt.savefig(image_name)
    #precision, recall, threshold = precision_recall_curve(y_test, y_pred)
    '''

def test_SVM_RandomForest_KNN(X_test,y_test):
    names = ["SVMRBF","Random Forest","KNN"]
    clfs = [_joblib.load('SVMRBF.joblib.pkl'),_joblib.load('RandomForest.joblib.pkl'),_joblib.load('KNN.joblib.pkl')]
    scalers = [_joblib.load('SVMRBFTrainScalar.joblib.pkl'),_joblib.load('RandomForestTrainScalar.joblib.pkl'),_joblib.load('KNNTrainScalar.joblib.pkl')]
    x = []
    avg =[]
    my_dict_avg = {}
    for name, clf, scaler in zip(names, clfs, scalers):
    
        X_testScaled = scaler.transform(X_test)
        y_pred = clf.predict(X_testScaled)
        print('{} percision: '.format(name),metrics.precision_score(y_test, y_pred,average="micro"))
        print('{} accuracy: '.format(name),metrics.accuracy_score(y_test, y_pred))
        class_report_dict = classification_report(y_test, y_pred,output_dict=True)
        for key,val in class_report_dict.items():
            try:
                float(key)
                if  not key in my_dict_avg:
                    x.append(key)
                    my_dict_avg[key] = {'precision':[val['precision']],'recall':[val['recall']],'f1-score':[val['f1-score']]}
                else:
                    my_dict_avg[key]['precision'].append(val['precision'])
                    my_dict_avg[key]['recall'].append(val['recall'])
                    my_dict_avg[key]['f1-score'].append(val['f1-score'])
                    #my_dict_avg[key]['accuracy'].append(val['accuracy'])
            except:
                continue
        avg.append(metrics.accuracy_score(y_test, y_pred))
    
    y = []
    i = 0
    x = ['among_us', 'facebook', 'instagram', 'pubg','telegram','tien_len']
    
    #x = ['dropbox', 'facebook', 'google', 'microsoft', 'teamviewer', 'twitter', 'youtube', 'unknown']
    #x = ['Linux', 'Windows', 'OSX']
    
    fo = open("here.txt", "w")
    fo.write('###########################BOA models#######################################'+'\n')
    fo.write('avg for each classifiers on: precision, recall, f1-score for each class.'+'\n')
    fo.write("classifiers names =  KNN, RBF SVM, Random Forest."+'\n')
    fo.write('class         '+' precision '+' recall '+' f1-score '+'\n')
    for key,val in my_dict_avg.items():
        f1 = np.average(val['f1-score'])
        p = np.average(val['precision'])
        r = np.average(val['recall'])
        fo.write(x[i]+'         '+' '+str(p)+' '+' '+str(r)+' '+' '+str(f1)+' '+'\n')
        print("label f1-score precision recall: ",x[i]+' '+str(f1)+' '+str(p)+' '+str(r))
        y.append(np.average(val['f1-score']))
        i= i+1
    
    fo.write('accuracy avg for all classifiers: '+str(np.average(avg)))
    fo.close()
    
    print("unsorted: ",y)
    unsorted_list = [(acc,label) for acc,label in zip(y,x)]
    sorted_list = sorted(unsorted_list,reverse=True)
    
    features_sorted = []
    importance_sorted = []
    for i in sorted_list:
        features_sorted += [i[1]]
        importance_sorted += [i[0]]
    
    plt.figure(num=None, figsize=(20,18), dpi=120, facecolor='w', edgecolor='r')

    sns.barplot(x=importance_sorted,y=features_sorted,dodge=False)
    plt.xlabel('score')
    plt.ylabel("class") 
    plt.show()

def plot_confusion_matrix(cm, eunique_label,title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(eunique_label))
    plt.xticks(tick_marks, eunique_label, rotation=45)
    plt.yticks(tick_marks, eunique_label)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def trainTestM(X_train,X_test,y_train,y_test):
    h = .02  # step size in the mesh
    my_dict_avg ={}
    names = [ "Nearest Neighbors", "Linear SVM", "RBF SVM","Decision Tree",
         "Random Forest", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
    classifiers = [
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=128),
    SVC(gamma=0.5, C=128),
    DecisionTreeClassifier(max_depth=15),
    # DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=15, n_estimators=10, max_features=10),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
    my_dict_avg ={}
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #y_test = transform_to_browser_labels(copy.deepcopy(y_test))
    #y_train = transform_to_browser_labels(copy.deepcopy(y_train))
    # iterate over classifiers
    x = []
    avg = []
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        print('name: ',name,'score: ',score)
        print(name,' percision: ',metrics.precision_score(y_test, y_pred,average="micro"))
        print(name,' accuracy: ',metrics.accuracy_score(y_test, y_pred,))
        class_report  = classification_report(y_test, y_pred,zero_division = "warn")
        class_report_dict = classification_report(y_test, y_pred,output_dict=True)
        for key,val in class_report_dict.items():
            try:
                float(key)
                if  not key in my_dict_avg:
                    x.append(key)
                    my_dict_avg[key] = {'precision':[val['precision']],'recall':[val['recall']],'f1-score':[val['f1-score']]}
                else:
                    my_dict_avg[key]['precision'].append(val['precision'])
                    my_dict_avg[key]['recall'].append(val['recall'])
                    my_dict_avg[key]['f1-score'].append(val['f1-score'])
                    #my_dict_avg[key]['accuracy'].append(val['accuracy'])
            except:
                continue
        #print()
        out_name= 'accuracy_report_{}_.txt'.format(name)#,target_names=target_names
        with open(out_name, "w") as text_file:
            text_file.write(class_report)
            text_file.write('percision: %s'%metrics.precision_score(y_test, y_pred,average="micro"))
            text_file.write(' accuracy: %s'%metrics.accuracy_score(y_test, y_pred))
        print(class_report)
        avg.append(metrics.accuracy_score(y_test, y_pred))
        print("my dict avg: ",my_dict_avg)
        filename = "clas_{}".format(name)
        _joblib.dump(clf, filename, compress=9)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
        print(cm_normalized)
        plt.figure()
        eunique_label = np.unique(y_train)#.tolist()
        plot_confusion_matrix(cm_normalized, eunique_label,title='Normalized confusion matrix')
        image_name = name + ".pdf"
        plt.savefig(image_name)
     
    print("my dict avg: ",my_dict_avg)
    '''
    with open('all_labels.csv', mode='r') as inp:
        reader = csv.reader(inp)
        dict_from_csv = {rows[4]:(rows[1],rows[2],rows[3]) for rows in reader}
    
    for key,val in dict_from_csv.items():
        if key.isnumeric():
            trance_key = float(key)
            cheak_key_app = int((trance_key % 10000) / 1000)*1000
            cheak_key_browser = int((trance_key % 1000) / 100)*100
            if cheak_key_app==8000:
                tup = (dict_from_csv[key][0],dict_from_csv[key][1],'unknown')
                dict_from_csv[key] = tup
            if cheak_key_browser==600:
                tup = (dict_from_csv[key][0],'Non-browser',dict_from_csv[key][2])
                dict_from_csv[key] = tup
            if cheak_key_app==8000 and cheak_key_browser==600:
                tup = (dict_from_csv[key][0],'Non-browser','unknown')
                dict_from_csv[key] = tup
    
    print()
    print("dict_from_csv: ",dict_from_csv)
    print
    
    #fo = open("all_label.txt", "w")
    my_label = []
    for i in x:
        
        if i in dict_from_csv:
            #fo.write(i + ' = ' + str(dict_from_csv[i]) + '\n')
            t = ' '.join(dict_from_csv[i])
            my_label.append(t)
    #fo.close()
    print("my_label: ",my_label)
    '''
    y = []
    res = {}
    i = 0
    x = ['among_us', 'facebook', 'instagram', 'pubg','telegram','tien_len']
    #x = ['dropbox', 'facebook', 'google', 'microsoft', 'teamviewer', 'twitter', 'youtube', 'unknown']
    #x = ['Linux', 'Windows', 'OSX']
    
    fo = open("here.txt", "w")
    fo.write('###########################sklearn models#######################################'+'\n')
    fo.write('avg for each classifiers on: precision, recall, f1-score for each class.'+'\n')
    fo.write("classifiers names =  Nearest Neighbors, Linear SVM, RBF SVM, Decision Tree, Random Forest, AdaBoost, Naive Bayes, Linear Discriminant Analysis, Quadratic Discriminant Analysis"+'\n')
    fo.write('class         '+' precision '+' recall '+' f1-score '+'\n')
    for key,val in my_dict_avg.items():
        f1 = np.average(val['f1-score'])
        p = np.average(val['precision'])
        r = np.average(val['recall'])
        fo.write(x[i]+'         '+' '+str(p)+' '+' '+str(r)+' '+' '+str(f1)+' '+'\n')
        print("label f1-score precision recall: ",x[i]+' '+str(f1)+' '+str(p)+' '+str(r))
        y.append(np.average(val['f1-score']))
        i= i+1
    
    fo.write('accuracy avg for all classifiers: '+str(np.average(avg)))
    fo.close()
    
    print("unsorted: ",y)
    unsorted_list = [(acc,label) for acc,label in zip(y,x)]
    sorted_list = sorted(unsorted_list,reverse=True)
    
    features_sorted = []
    importance_sorted = []
    for i in sorted_list:
        features_sorted += [i[1]]
        importance_sorted += [i[0]]
    
    plt.figure(num=None, figsize=(20,18), dpi=120, facecolor='w', edgecolor='r')

    sns.barplot(x=importance_sorted,y=features_sorted,dodge=False)
    plt.xlabel('score')
    plt.ylabel("class") 
    plt.show()
    
def ensembleVoting(X_train,y_train,X_test,y_test):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # num_folds = 3
    # num_instances = len(X)
    # seed = 7
    # kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    # create the sub models
    #estimators = []
    # model1 = LogisticRegression()
    # estimators.append(('logistic', model1))
    # model2 = DecisionTreeClassifier()
    # estimators.append(('cart', model2))
    # model3 = SVC()
    # estimators.append(('svm', model2))
    # names = [ "Nearest Neighbors", "Linear SVM", "RBF SVM","Decision Tree",
    #      "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
    #      "Quadratic Discriminant Analysis"]

    #model1 = KNeighborsClassifier()
    #estimators.append(('knn', model1))
    #model2 = SVC()
    #estimators.append(('svmrbf', model2))
    #model3 = DecisionTreeClassifier(max_depth=20)
    #estimators.append(('DecisionTree', model3))
    # model4 = LinearDiscriminantAnalysis()
    # estimators.append(('LDA', model4))

    classifiers =  [
    ('Nearest Neighbors',KNeighborsClassifier(32)),
    ('Linear SVM',SVC(kernel="linear", C=0.025)),
    ('RBF SVM',SVC(gamma=2, C=1)),
    ('Decision Tree',DecisionTreeClassifier(max_depth=5)),
    # DecisionTreeClassifier(max_depth=5),
    ('Random Forest',RandomForestClassifier(max_depth=5, n_estimators=10, max_features=69)),
    ('AdaBoost',AdaBoostClassifier()),
    ('Naive Bayes',GaussianNB()),
    ('Linear Discriminant Analysis',LinearDiscriminantAnalysis()),
    ('Quadratic Discriminant Analysis',QuadraticDiscriminantAnalysis())]
    name = "EnsembleVoting"
    # create the ensemble model
    ensemble = VotingClassifier(classifiers)#, voting='soft', weights=[1,2,1,1])
    ensemble.fit(X_train,y_train)
    score = ensemble.score(X_test,y_test)
    print("score: ",score)
    y_pred = ensemble.predict(X_test)
    print(y_pred)
    class_report  = classification_report(y_test, y_pred)
    out_name= 'accuracy_report_{}_.txt'.format(name)#,target_names=target_names
    with open(out_name, "w") as text_file:
        text_file.write(class_report)
        text_file.write('percision: %s'%metrics.precision_score(y_test, y_pred,average="micro"))
        text_file.write(' accuracy: %s'%metrics.accuracy_score(y_test, y_pred))
    

    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    eunique_label = np.unique(y_train)#.tolist()
    plot_confusion_matrix(cm_normalized, eunique_label,title='Normalized confusion matrix')
    image_name = name + ".pdf"
    plt.savefig(image_name)
    '''
    params = {'svmrbf__gamma': [2**(-7),2**(-5),2**(-3)],
              'svmrbf__C': [2**(5),2**(7),2**(9),2**(11),2**(13)],
              'knn__algorithm': ['ball_tree'],
              'knn__n_neighbors': [14, 16, 20],
              'knn__metric': [ # 'chebyshev', 'sokalmichener',
              'canberra'#, 'dice', 'euclidean',
              #'braycurtis', 'russellrao','cityblock', 'manhattan']}
              ]}
    ensemble
    grid = GridSearchCV(estimator=ensemble, param_grid=params, n_jobs=-1)
    grid = grid.fit(X_train,y_train)
    print(grid.score(X_train,y_train))
    print(ensemble.score(X_test,y_test))
    results = cross_val_score(ensemble, X, y)
    print(results.mean())
    '''

def CategoricalEnsembleVoting(X_train,y_train,X_test,y_test):
    scaler = preprocessing.StandardScaler().fit(X_train)
    name = "CategoricalEnsembleVoting"
    print("X_test: ",X_test)
    
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = CategoryClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.score(X_test, y_test)
    #print(repr(clf.score(X_test, y_test)))
    print("y_pred: ",y_pred)
    print("y_test: ",y_test)
    class_report  = classification_report(y_test, y_pred)
    out_name= 'accuracy_report_{}_.txt'.format(name)#,target_names=target_names
    with open(out_name, "w") as text_file:
        text_file.write(class_report)
        text_file.write('percision: %s'%metrics.precision_score(y_test, y_pred,average="micro"))
        text_file.write(' accuracy: %s'%metrics.accuracy_score(y_test, y_pred))
    print(class_report)
    
    y_pred = clf.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    eunique_label = np.unique(y_train)#.tolist()
    plot_confusion_matrix(cm_normalized, eunique_label,title='Normalized confusion matrix')
    image_name = name + ".pdf"
    plt.savefig(image_name)
    
    

""" mean: 0.99819, std: 0.00051, params:
    {'svmrbf__gamma': 0.0078125, 'knn__algorithm': 'ball_tree',
    'knn__n_neighbors': 16, 'svmrbf__C': 8192, 'knn__metric': 'canberra'} """
def fiveFold():

    # Feature groups
    # protocol_dependent = range(13) + range(66,69)
    # protocol_dependent = range(23) + range(66,69)
    # peak features
    # protocol_dependent = range(23,41)
    # All but peak
    # protocol_dependent = range(23) + range(41,69)
    fsslv_cipher_suites = [6,7,8,9,10,11,12]
    protocol_dependent = []

    # Load data
    data_path = os.getcwd() + "/data_set/libSVM"
    print(data_path)

    train_0 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_0_train"
    test_0 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_0_test"
    train_1 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_1_train"
    test_1 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_1_test"
    train_2 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_2_train"
    test_2 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_2_test"
    train_3 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_3_train"
    test_3 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_3_test"
    train_4 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_4_train"
    test_4 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_4_test"

    X_train_0, y_train_0, X_test_0, y_test_0 = load_svmlight_files(
    (train_0, test_0))
    X_train_1, y_train_1, X_test_1, y_test_1 = load_svmlight_files(
    (train_1, test_1))
    X_train_2, y_train_2, X_test_2, y_test_2 = load_svmlight_files(
    (train_2, test_2))
    X_train_3, y_train_3, X_test_3, y_test_3 = load_svmlight_files(
    (train_3, test_3))
    X_train_4, y_train_4, X_test_4, y_test_4 = load_svmlight_files(
    (train_4, test_4))

    df_train_0 = pd.DataFrame(X_train_0.toarray())
    df_test_0 = pd.DataFrame(X_test_0.toarray())
    df_train_1 = pd.DataFrame(X_train_1.toarray())
    df_test_1 = pd.DataFrame(X_test_1.toarray())
    df_train_2 = pd.DataFrame(X_train_2.toarray())
    df_test_2 = pd.DataFrame(X_test_2.toarray())
    df_train_3 = pd.DataFrame(X_train_3.toarray())
    df_test_3 = pd.DataFrame(X_test_3.toarray())
    df_train_4 = pd.DataFrame(X_train_4.toarray())
    df_test_4 = pd.DataFrame(X_test_4.toarray())

    X_train_0 = df_train_0.drop(protocol_dependent, axis=1)
    X_test_0 = df_test_0.drop(protocol_dependent, axis=1)
    X_train_1 = df_train_1.drop(protocol_dependent, axis=1)
    X_test_1 = df_test_1.drop(protocol_dependent, axis=1)
    X_train_2 = df_train_2.drop(protocol_dependent, axis=1)
    X_test_2 = df_test_2.drop(protocol_dependent, axis=1)
    X_train_3 = df_train_3.drop(protocol_dependent, axis=1)
    X_test_3 = df_test_3.drop(protocol_dependent, axis=1)
    X_train_4 = df_train_4.drop(protocol_dependent, axis=1)
    X_test_4 = df_test_4.drop(protocol_dependent, axis=1)

    # X_train_0 = randomProtocolValues(X_train_0)
    # X_test_0 = randomProtocolValues(X_test_0)
    # X_train_1 = randomProtocolValues(X_train_1)
    # X_test_1 = randomProtocolValues(X_test_1)
    # X_train_2 = randomProtocolValues(X_train_2)
    # X_test_2 = randomProtocolValues(X_test_2)
    # X_train_3 = randomProtocolValues(X_train_3)
    # X_test_3 = randomProtocolValues(X_test_3)
    # X_train_4 = randomProtocolValues(X_train_4)
    # X_test_4 = randomProtocolValues(X_test_4)


    # Prepare ensemble method
    estimators = []
    model1 = KNeighborsClassifier(n_neighbors=16,algorithm='ball_tree',
                                    metric='canberra', n_jobs=-1)
    estimators.append(('knn', model1))
    model2 = SVC(gamma=0.0078125,C=8192, probability=False)
    estimators.append(('svmrbf', model2))
    model3 = DecisionTreeClassifier()#max_depth=50)
    estimators.append(('DecisionTree', model3))
    model4 = RandomForestClassifier(n_estimators=100, oob_score=True,
                                    n_jobs=-1)
    estimators.append(('RandomForest', model4))
    model5 = XGBClassifier(max_depth=10, n_estimators=100, learning_rate=0.1)
    estimators.append(('XGBoost', model5))

    # ensemble = VotingClassifier(estimators,voting='hard')
    ensemble = CategoryClassifier()

    # CategoricalEnsembleVoting(X_train_0, y_train_0, X_test_0, y_test_0)
    oneFold(X_train_0, y_train_0, X_test_0, y_test_0, ensemble)
    oneFold(X_train_1, y_train_1, X_test_1, y_test_1, ensemble)
    oneFold(X_train_2, y_train_2, X_test_2, y_test_2, ensemble)
    oneFold(X_train_3, y_train_3, X_test_3, y_test_3, ensemble)
    oneFold(X_train_4, y_train_4, X_test_4, y_test_4, ensemble)


def randomProtocolValues(X):
    # protocol features
    a = range(13) + range(66,69)
    l = len(X)

    for i in a:
        X[i] = np.random.random(l) * 100

    return X



def oneFold(X_train,y_train,X_test,y_test,clf):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf.fit(X_train, y_train)
    print(repr(clf.score(X_test, y_test)))

#-----------------------MAIN------------------------------------------------------
if __name__ == "__main__":

    data_path = os.getcwd() + "/data_set"
    #all_features_path = data_path + "/samples_tab.csv"
    all_features_path = data_path + "/samples.csv"
    rows_to_skip = [0]
    ds = read_data_set(all_features_path, rows_to_skip=rows_to_skip)
    #ds = ds.drop(10187)
    #ds = ds.drop(10288)
    print(repr(ds)) 

    ds = ds.dropna()
    #print("DS: ",repr(ds))
    y = ds.iloc[:,len(ds.columns)-1]
    #print("label: ",y)
    # num_classes, y = np.unique(y, return_inverse=True)
    # X = ds.drop([str(len(ds.columns)-1)], axis=1)
    X = ds.drop([len(ds.columns)-1], axis=1)
    #print("X: ",X)
    # X = ds.drop([2,4,9,11], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print("X_train : ",X_train)
    print("y_train : ",y_train)
    print("X_test : ",X_test)
    print("y_test : ",y_test)

    


    #model1 = SVC()
    #model1.fit(X_train,y_train)
    #print(model1.score(X_test,y_test))
    #y_train = transform_to_app_labels(copy.deepcopy(y_train))
    #y_test = transform_to_app_labels(copy.deepcopy(y_test))
    
    #classes, y_indices = np.unique(y_train, return_inverse=True)
    #print(classes)
    #print(y_indices)
    #class_counts = np.bincount(y_indices)
    #print(class_counts)

    #trainTestM(X_train,X_test,y_train,y_test)
    
    #model = RandomForestClassifier(n_estimators=100)
    #model.fit(X_train,y_train)
    #print(model.score(X_test,y_test))
    
    trainRandomForest(X_train,y_train)
    testRandomForest(X_test,y_test)
    trainKNN(X_train,y_train)
    testKNN(X_test,y_test)
    trainSVC_RBF(X_train,y_train)
    testSVC_RBF(X_test,y_test)
    
    test_SVM_RandomForest_KNN(X_test,y_test)
    
    #ensembleVoting(X_train, y_train, X_test, y_test)
    #CategoricalEnsembleVoting(X_train, y_train, X_test, y_test)
    #fiveFold()
