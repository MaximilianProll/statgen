import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

from sklearn.model_selection import KFold
import matplotlib.patches as mpatches


# Importing the data
data = pd.read_csv("neonatal_basic_data.csv")
training_set = data[data['death']!=0]
training_labels = training_set['death']
training_data = training_set[['ga', 'bw']]
test_set = data[data['death']==0]
#test_labels = test_set['death']
test_data = test_set[['ga', 'bw']]


# ### Scaling the data
scaler = StandardScaler()
training_data_stand = scaler.fit_transform(training_data)
test_data_stand = scaler.fit_transform(test_data)

minmaxscaler = MinMaxScaler()
training_data_minmax = minmaxscaler.fit_transform(training_data)
test_data_minmax = minmaxscaler.fit_transform(test_data)


# ## Visualization of the class distribution
labels = training_labels.as_matrix()
classes, distribution = np.unique(labels, return_counts=True)
plt.bar(classes, distribution)
plt.xticks(classes, ["survived", "died"])
plt.show()

sns.pairplot(training_set[['ga', 'bw', 'death']]
             , hue="death"
             , vars=['ga', 'bw']
             , diag_kind="kde"
             #, kind="reg"
             , size=3
            )
plt.show()

#kernel = 1.0 * C(constant_value=1.0) + 1.0 * RBF(length_scale=1.0)
kernel = 1.0 * RBF(length_scale=1.0)
clfs = [(GaussianProcessClassifier(kernel=kernel, optimizer=None),"Gaussian Processes(RBF)")
        , (svm.SVC(class_weight='balanced', kernel='rbf'),"SVM")
        , (DecisionTreeClassifier(),"Decision tree")
        , (LogisticRegression(),"Logistic regression")
        , (GaussianNB(),"Gaussian naive Bayes")
       ]

for clf, method_name in clfs:
    misclassif_lst = []
    precision_lst = []
    accuracy_lst = []
    recall_lst = []
    f1_lst = []

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(training_data_stand):
        X_train, X_test = training_data_stand[train_index], training_data_stand[test_index]
        y_train, y_test = training_labels[train_index], training_labels[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        misclassif = np.sum(y_pred!=y_test)
        precision = precision_score(y_pred, y_test)
        accuracy = accuracy_score(y_pred, y_test) 
        recall = recall_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test)

        misclassif_lst.append(misclassif)
        precision_lst.append(precision)
        accuracy_lst.append(accuracy)
        recall_lst.append(recall)

        f1_lst.append(f1)

    plt.plot(range(5), precision_lst, label="Precision")
    plt.plot(range(5), accuracy_lst, label="Accuracy")
    plt.plot(range(5), recall_lst, label="Recall")
    plt.plot(range(5), f1_lst, label="F1-score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(method_name)
    plt.show()

    print("Average on k-fold cross validation")
    print("misclassif error:", np.mean(misclassif_lst))
    print("precision:", np.mean(precision_lst))
    print("accuracy:", np.mean(accuracy_lst))
    print("recall:", np.mean(recall_lst))
    print("f1:", np.mean(f1_lst))

    clf.fit(training_data_stand, training_labels)
    p = clf.predict(test_data_stand)

    classes, distribution = np.unique(p, return_counts=True)
    plt.bar(classes, distribution)
    plt.title(method_name)
    plt.xticks(classes, ["survived", "died"])
    plt.show()

    pred = pd.DataFrame(data={'ga':test_set['ga'].as_matrix()
    	, 'bw':test_set['bw'].as_matrix(), 'death':p})

    if method_name=="Gaussian Processes(RBF)":
        gp_pred = pred
    elif method_name=="SVM":
        svm_pred = pred    
    elif method_name=="Decision tree":
        dt_pred = pred
    elif method_name=="Logistic regression":
        lr_pred = pred
    elif method_name=="Gaussian naive Bayes":
        gnb_pred = pred

    sns.pairplot(pred[['ga', 'bw', 'death']]
                 , hue="death"
                 , vars=['ga', 'bw']
                 #, diag_kind="kde"
                 #, kind="reg"
                 , size=3
                )
    plt.title(method_name)
    plt.show()


# Mixing prediction
# Final prediction are:
#     - Survived: if all the models agree that the patient survived
#     - Died: if at least one of the model classify the patient as died

pred = pd.DataFrame(data={'patientid':test_set['patientid'].as_matrix()
	, 'ga':test_set['ga'].as_matrix(), 'bw':test_set['bw'].as_matrix()
    , 'death_dt':dt_pred['death'], 'death_svm':svm_pred['death']
    , 'death_lr':lr_pred['death']
    , 'death_gnb':gnb_pred['death'], 'death_gp':gp_pred['death']})

died = pred.loc[(pred['death_lr'] == 2) | (pred['death_dt'] == 2) 
	| (pred['death_svm']==2) | (pred['death_gp'] == 2) 
	| (pred['death_gnb']==2)]
survived = pred.loc[(pred['death_lr'] == 1) & (pred['death_dt'] == 1) 
	& (pred['death_svm']==1) & (pred['death_gp'] == 1) 
	& (pred['death_gnb'] == 1)]
plt.figure(figsize=(8,6))
plt.scatter(survived['ga'], survived['bw'])
plt.scatter(died['ga'], died['bw'])
plt.xlabel("ga", fontsize=14)
plt.ylabel("bw", fontsize=14)

orange_patch = mpatches.Patch(color='orange', label='1 - died')
blue_patch = mpatches.Patch(color='blue', label='2 - survived')
plt.legend(handles=[orange_patch, blue_patch])
plt.title("Basic data - combined predictions", fontsize=14)
plt.show()


# ## Time-series data classification
ts = pd.read_csv("timeseries.csv", names=['patientid','mean ABP_S','var ABP_S'
	,'slope ABP_S','intercept ABP_S','mean ABP_M','var ABP_M','slope ABP_M'
	,'intercept ABP_M','mean ABP_D','var ABP_D','slope ABP_D','intercept ABP_D'
	,'mean HR_ECG','var HR_ECG','slope HR_ECG','intercept HR_ECG','mean SpO2'
	,'var SpO2','slope SpO2','intercept SpO2'])
training_data = ts[ts['patientid'].isin(data[data['death']!=0]['patientid'])]
training_data.drop(['patientid'], axis=1)
test_data = ts[ts['patientid'].isin(data[data['death']==0]['patientid'])]
test_data.drop(['patientid'], axis=1)
ts.shape

scaler = StandardScaler()
training_data_stand = scaler.fit_transform(training_data)
test_data_stand = scaler.fit_transform(test_data)

minmaxscaler = MinMaxScaler()
training_data_minmax = minmaxscaler.fit_transform(training_data)
test_data_minmax = minmaxscaler.fit_transform(test_data)

# Evaluation of classifier via K-fold cross-validation
for clf, method_name in clfs:
    misclassif_lst = []
    precision_lst = []
    accuracy_lst = []
    recall_lst = []
    f1_lst = []

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(training_data_stand):
        X_train, X_test = training_data_stand[train_index], training_data_stand[test_index]
        y_train, y_test = training_labels[train_index], training_labels[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        misclassif = np.sum(y_pred!=y_test)
        precision = precision_score(y_pred, y_test)
        accuracy = accuracy_score(y_pred, y_test) 
        recall = recall_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test)

        misclassif_lst.append(misclassif)
        precision_lst.append(precision)
        accuracy_lst.append(accuracy)
        recall_lst.append(recall)

        f1_lst.append(f1)

    plt.plot(range(5), precision_lst, label="Precision")
    plt.plot(range(5), accuracy_lst, label="Accuracy")
    plt.plot(range(5), recall_lst, label="Recall")
    plt.plot(range(5), f1_lst, label="F1-score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(method_name)
    plt.show()

    print("Average on k-fold cross validation")
    print("misclassif error:", np.mean(misclassif_lst))
    print("precision:", np.mean(precision_lst))
    print("accuracy:", np.mean(accuracy_lst))
    print("recall:", np.mean(recall_lst))
    print("f1:", np.mean(f1_lst))

    clf.fit(training_data_stand, training_labels)
    p = clf.predict(test_data_stand)

    classes, distribution = np.unique(p, return_counts=True)
    plt.bar(classes, distribution)
    plt.title(method_name)
    plt.xticks(classes, ["survived", "died"])
    plt.show()

    pred = pd.DataFrame(data={'ga':test_set['ga'].as_matrix()
    	, 'bw':test_set['bw'].as_matrix(), 'death':p})
    
    if method_name=="Gaussian Processes(RBF)":
        gp_pred = pred
    elif method_name=="SVM":
        svm_pred = pred    
    elif method_name=="Decision tree":
        dt_pred = pred
    elif method_name=="Logistic regression":
        lr_pred = pred
    #elif method_name=="Gaussian naive Bayes":
    #    gnb_pred = pred
    
    sns.pairplot(pred[['ga', 'bw', 'death']]
                 , hue="death"
                 , vars=['ga', 'bw']
                 , diag_kind="kde"
                 #, kind="reg"
                 , size=3
                )
    plt.title(method_name)
    plt.show()

pred = pd.DataFrame(data={'patientid':test_set['patientid'].as_matrix()
	, 'ga':test_set['ga'].as_matrix(), 'bw':test_set['bw'].as_matrix()
    , 'death_dt':dt_pred['death'], 'death_svm':svm_pred['death']
    , 'death_lr':lr_pred['death'], 'death_gp':gp_pred['death']})

died = pred.loc[(pred['death_dt'] == 2) | (pred['death_svm']==2) 
	| (pred['death_gp'] == 2) | (pred['death_lr']==2)]
survived = pred.loc[(pred['death_dt'] == 1) & (pred['death_svm']==1) 
	& (pred['death_gp'] == 1) & (pred['death_lr'] == 1)]
plt.figure(figsize=(8,6))
plt.scatter(survived['ga'], survived['bw'])
plt.scatter(died['ga'], died['bw'])
plt.xlabel("ga", fontsize=14)
plt.ylabel("bw", fontsize=14)

orange_patch = mpatches.Patch(color='orange', label='1 - died')
blue_patch = mpatches.Patch(color='blue', label='2 - survived')
plt.legend(handles=[orange_patch, blue_patch])
plt.title("Time-series data - combined predictions", fontsize=14)
plt.show()


# Combining time-series with basic data
# ts, data
fulldata = data.merge(ts, on='patientid')
training_set = fulldata[fulldata['death']!=0]
training_labels = training_set['death']
training_data = training_set.drop(['patientid', 'death'], axis=1)
test_set = fulldata[fulldata['death']==0]
test_data = test_set.drop(['patientid', 'death'], axis=1)

scaler = StandardScaler()
training_data_stand = scaler.fit_transform(training_data)
test_data_stand = scaler.fit_transform(test_data)

minmaxscaler = MinMaxScaler()
training_data_minmax = minmaxscaler.fit_transform(training_data)
test_data_minmax = minmaxscaler.fit_transform(test_data)

for clf, method_name in clfs:
    misclassif_lst = []
    precision_lst = []
    accuracy_lst = []
    recall_lst = []
    f1_lst = []

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(training_data_stand):
        X_train, X_test = training_data_stand[train_index], training_data_stand[test_index]
        y_train, y_test = training_labels[train_index], training_labels[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        misclassif = np.sum(y_pred!=y_test)
        precision = precision_score(y_pred, y_test)
        accuracy = accuracy_score(y_pred, y_test) 
        recall = recall_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test)

        misclassif_lst.append(misclassif)
        precision_lst.append(precision)
        accuracy_lst.append(accuracy)
        recall_lst.append(recall)

        f1_lst.append(f1)

    plt.plot(range(5), precision_lst, label="Precision")
    plt.plot(range(5), accuracy_lst, label="Accuracy")
    plt.plot(range(5), recall_lst, label="Recall")
    plt.plot(range(5), f1_lst, label="F1-score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(method_name)
    plt.show()

    print("Average on k-fold cross validation")
    print("misclassif error:", np.mean(misclassif_lst))
    print("precision:", np.mean(precision_lst))
    print("accuracy:", np.mean(accuracy_lst))
    print("recall:", np.mean(recall_lst))
    print("f1:", np.mean(f1_lst))

    clf.fit(training_data_stand, training_labels)
    p = clf.predict(test_data_stand)

    classes, distribution = np.unique(p, return_counts=True)
    plt.bar(classes, distribution)
    plt.title(method_name)
    plt.xticks(classes, ["survived", "died"])
    plt.show()

    pred = pd.DataFrame(data={'ga':test_set['ga'].as_matrix()
    	, 'bw':test_set['bw'].as_matrix(), 'death':p})
    
    if method_name=="Gaussian Processes(RBF)":
        gp_pred = pred
    elif method_name=="SVM":
        svm_pred = pred    
    elif method_name=="Decision tree":
        dt_pred = pred
    elif method_name=="Logistic regression":
        lr_pred = pred
    #elif method_name=="Gaussian naive Bayes":
    #    gnb_pred = pred
    
    sns.pairplot(pred[['ga', 'bw', 'death']]
                 , hue="death"
                 , vars=['ga', 'bw']
                 , diag_kind="kde"
                 #, kind="reg"
                 , size=3
                )
    plt.title(method_name)
    plt.show()

pred = pd.DataFrame(data={'patientid':test_set['patientid'].as_matrix()
    , 'ga':test_set['ga'].as_matrix(), 'bw':test_set['bw'].as_matrix()
    , 'death_dt':dt_pred['death'], 'death_svm':svm_pred['death']
    , 'death_lr':lr_pred['death'], 'death_gp':gp_pred['death']})

died = pred.loc[(pred['death_dt'] == 2) | (pred['death_svm']==2) 
	| (pred['death_gp'] == 2) | (pred['death_lr']==2)]
survived = pred.loc[(pred['death_dt'] == 1) & (pred['death_svm']==1) 
	& (pred['death_gp'] == 1) & (pred['death_lr'] == 1)]
plt.figure(figsize=(8,6))
plt.scatter(survived['ga'], survived['bw'])
plt.scatter(died['ga'], died['bw'])
plt.xlabel("ga", fontsize=14)
plt.ylabel("bw", fontsize=14)

orange_patch = mpatches.Patch(color='orange', label='1 - died')
blue_patch = mpatches.Patch(color='blue', label='2 - survived')
plt.legend(handles=[orange_patch, blue_patch])
plt.title("Combined data - combined predictions", fontsize=14)
plt.show()

died["patientid"].as_matrix()