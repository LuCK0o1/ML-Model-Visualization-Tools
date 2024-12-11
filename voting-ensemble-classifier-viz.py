import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import accuracy_score
# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier , RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def load_initial_graph(data , ax):
    if data == 'Circle':
        csv = Path("C:/Users/SHUBHAM/PycharmProjects/ML practice/Data/circle.csv")
        df = pd.read_csv(csv.resolve() , index_col=0)
        X = np.array(df.iloc[:, 0:2])
        y = np.array(df.iloc[:, 2])
    elif data == 'Blob':
        csv = Path("C:/Users/SHUBHAM/PycharmProjects/ML practice/Data/blob.csv")
        df = pd.read_csv(csv.resolve() , index_col=0)
        X = np.array(df.iloc[:, 0:2])
        y = np.array(df.iloc[:, 2])
    elif data == 'Moons':
        csv = Path("C:/Users/SHUBHAM/PycharmProjects/ML practice/Data/moons.csv")
        df = pd.read_csv(csv.resolve() , index_col=0)
        X = np.array(df.iloc[:, 0:2])
        y = np.array(df.iloc[:, 2])
    elif data == 'Guassiun':
        csv = Path("C:/Users/SHUBHAM/PycharmProjects/ML practice/Data/guassian.csv")
        df = pd.read_csv(csv.resolve() , index_col=0)
        X = np.array(df.iloc[:, 0:2])
        y = np.array(df.iloc[:, 2])
    ax.scatter(X[:,0].T, X[:,1].T, c=y, cmap='rainbow')

    return X, y

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array


plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Voting Ensemble Classifier")

data = st.sidebar.selectbox(
    'Data',
    ('Circle','Blob','Moons','Guassiun')
)

models = st.sidebar.multiselect(
    'Models : (5-SVM / Others) at a time',
    ('Logistic Regression', 'SVC', 'Random Forest', 'Naive Byes', 'KNN' , '5-SVM')
)

voting = st.sidebar.selectbox(
    'Voting Type',
    ('soft', 'hard')
)

cols = len(models)
if '5-SVM' in models:
    cols += 4
elif cols == 0:
    cols = 1
# Load initial graph
fig, ax = plt.subplots()
figs, axs = plt.subplots(nrows=1,ncols=cols)

# Plot initial graph
X,y = load_initial_graph(data, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

# train initial models
estimators = []
#print(models)
for i in models:
    if i == 'Logistic Regression':
        lr = LogisticRegression()
        lr.fit(X_train , y_train)
        lr_pred = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)

        st.sidebar.text(f'Logistic Regression : {lr_acc}')

        estimators.append(('lr', lr))

    elif i == 'SVC':
        svc = SVC()
        svc.fit(X_train , y_train)
        svc_pred = svc.predict(X_test)
        svc_acc = accuracy_score(y_test , svc_pred)

        st.sidebar.text(f'SVC : {svc_acc}')

        estimators.append(('svc',svc))

    elif i == 'Random Forest':
        rf = RandomForestClassifier()
        rf.fit(X_train , y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test , rf_pred)

        st.sidebar.text(f'Random Forest : {rf_acc}')

        estimators.append(('rf',rf))

    elif i == 'Naive Byes':
        nb = GaussianNB()
        nb.fit(X_train , y_train)
        nb_pred = nb.predict(X_test)
        nb_acc = accuracy_score(y_test , nb_pred)

        st.sidebar.text(f'Naive Byes : {nb_acc}')

        estimators.append(('nb',nb))

    elif i == 'KNN':
        knn = KNeighborsClassifier()
        knn.fit(X_train , y_train)
        knn_pred = knn.predict(X_test)
        knn_acc = accuracy_score(y_test , knn_pred)

        st.sidebar.text(f'KNN : {knn_acc}')

        estimators.append(('knn',knn))

    elif i == '5-SVM':
        svm1 = SVC(probability=True, kernel='poly', degree=1)
        svm2 = SVC(probability=True, kernel='poly', degree=2)
        svm3 = SVC(probability=True, kernel='poly', degree=3)
        svm4 = SVC(probability=True, kernel='poly', degree=4)
        svm5 = SVC(probability=True, kernel='poly', degree=5)

        svm1.fit(X_train , y_train)
        svm2.fit(X_train, y_train)
        svm3.fit(X_train, y_train)
        svm4.fit(X_train, y_train)
        svm5.fit(X_train, y_train)

        estimators = [('SVM1 : ', svm1), ('SVM2 : ', svm2), ('SVM3 : ', svm3), ('SVM4 : ', svm4), ('SVM5 : ', svm5)]

        for estimator in estimators:
            acc = cross_val_score(estimator[1], X_train, y_train, cv=10, scoring='accuracy')
            st.sidebar.text(f'{estimator[0]}{np.round(np.mean(acc), 2)}')

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf = VotingClassifier(estimators=estimators,
                           voting=voting)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    #ax.xlabel("Col1")
    #ax.ylabel("Col2")
    orig = st.pyplot(fig)
    if '5-SVM' in models:
        st.subheader("Accuracy for Voting Ensemble Classifier  " + str(np.round(np.mean(cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')), 2)))
    else:
        st.subheader("Accuracy for Voting Ensemble Classifier  " + str(round(accuracy_score(y_test, y_pred), 2)))

    figs.set_figwidth(5)
    figs.set_figheight(1)
    ind = 0
    flag = 0
    while ind < len(models):
        i = models[ind]
        if 'Logistic Regression' == i:
            axs[ind-flag].contourf(XX, YY, lr.predict(input_array).reshape(XX.shape) , cmap='rainbow')
            axs[ind-flag].set_axis_off()
            axs[ind-flag].set_title('Logistic Regression' , fontsize=5)
        elif 'SVC' == i:
            axs[ind-flag].contourf(XX, YY, svc.predict(input_array).reshape(XX.shape), cmap='rainbow')
            axs[ind-flag].set_axis_off()
            axs[ind-flag].set_title('SVC' , fontsize=5)
        elif 'Random Forest' == i:
            axs[ind-flag].contourf(XX, YY, rf.predict(input_array).reshape(XX.shape), cmap='rainbow')
            axs[ind-flag].set_axis_off()
            axs[ind-flag].set_title('Random Forest' , fontsize=5)
        elif 'Naive Byes' == i:
            axs[ind-flag].contourf(XX, YY, nb.predict(input_array).reshape(XX.shape), cmap='rainbow')
            axs[ind-flag].set_axis_off()
            axs[ind-flag].set_title('Naive Bayes' , fontsize=5)
        elif 'KNN'== i:
            axs[ind-flag].contourf(XX, YY, knn.predict(input_array).reshape(XX.shape), cmap='rainbow')
            axs[ind-flag].set_axis_off()
            axs[ind-flag].set_title('KNN' , fontsize=5)
        elif '5-SVM' == i:
            flag = 1
        ind+=1
    ind-=1
    if flag == 1:
        axs[ind].contourf(XX, YY, svm1.predict(input_array).reshape(XX.shape), cmap='rainbow')
        axs[ind].set_axis_off()
        axs[ind].set_title('Degree-1', fontsize=5)

        axs[ind + 1].contourf(XX, YY, svm2.predict(input_array).reshape(XX.shape), cmap='rainbow')
        axs[ind + 1].set_axis_off()
        axs[ind + 1].set_title('Degree-2', fontsize=5)

        axs[ind + 2].contourf(XX, YY, svm3.predict(input_array).reshape(XX.shape), cmap='rainbow')
        axs[ind + 2].set_axis_off()
        axs[ind + 2].set_title('Degree-3', fontsize=5)

        axs[ind + 3].contourf(XX, YY, svm4.predict(input_array).reshape(XX.shape), cmap='rainbow')
        axs[ind + 3].set_axis_off()
        axs[ind + 3].set_title('Degree-4', fontsize=5)

        axs[ind + 4].contourf(XX, YY, svm5.predict(input_array).reshape(XX.shape), cmap='rainbow')
        axs[ind + 4].set_axis_off()
        axs[ind + 4].set_title('Degree-5', fontsize=5)


    st.pyplot(figs)

# open in pycharm or vs code
# after open terminal in that
# write command
# streamlit run voting-ensemble-classifier-viz.py
