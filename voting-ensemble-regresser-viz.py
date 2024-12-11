import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import r2_score as accuracy_score
# models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def load_initial_graph(data , ax):
    if data == 'Line':
        csv = Path("C:/Users/SHUBHAM/PycharmProjects/ML practice/Data/regg1.csv")
        df = pd.read_csv(csv.resolve() , index_col=0)
        X = np.array(df.iloc[:, 0]).reshape(-1,1)
        y = np.array(df.iloc[:, 1]).reshape(-1,1)
    elif data == 'S-Shape':
        csv = Path("C:/Users/SHUBHAM/PycharmProjects/ML practice/Data/regg2.csv")
        df = pd.read_csv(csv.resolve() , index_col=0)
        X = np.array(df.iloc[:, 0]).reshape(-1, 1)
        y = np.array(df.iloc[:, 1]).reshape(-1, 1)
    elif data == 'Sharp':
        csv = Path("C:/Users/SHUBHAM/PycharmProjects/ML practice/Data/regg3.csv")
        df = pd.read_csv(csv.resolve() , index_col=0)
        X = np.array(df.iloc[:, 0]).reshape(-1, 1)
        y = np.array(df.iloc[:, 1]).reshape(-1, 1)
    ax.scatter(X , y , c = 'b' , edgecolor = 'black')

    return X, y

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Voting Ensemble Classifier")

data = st.sidebar.selectbox(
    'Data',
    ('Line','S-Shape','Sharp')
)

models = st.sidebar.multiselect(
    'Models : (5-DTR / Others) at a time',
    ('Linear Regression', 'SVR', 'Decision Tree Regresser', 'KNN' , '5-SVM')
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
    if i == 'Linear Regression':
        lr = LinearRegression()
        lr.fit(X_train , y_train)
        lr_pred = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)

        st.sidebar.text(f'Linear Regression : {lr_acc}')

        estimators.append(('lr', lr))

    elif i == 'SVR':
        svc = SVR(degree=6)
        svc.fit(X_train , y_train)
        svc_pred = svc.predict(X_test)
        svc_acc = accuracy_score(y_test , svc_pred)

        st.sidebar.text(f'SVR : {svc_acc}')

        estimators.append(('svc',svc))

    elif i == 'Decision Tree Regresser':
        rf = DecisionTreeRegressor()
        rf.fit(X_train , y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test , rf_pred)

        st.sidebar.text(f'Random Forest : {rf_acc}')

        estimators.append(('rf',rf))

    elif i == 'KNN':
        knn = KNeighborsRegressor()
        knn.fit(X_train , y_train)
        knn_pred = knn.predict(X_test)
        knn_acc = accuracy_score(y_test , knn_pred)

        st.sidebar.text(f'KNN : {knn_acc}')

        estimators.append(('knn',knn))

    elif i == '5-SVM':
        svm1 = SVR(degree=1)
        svm2 = SVR(degree=2)
        svm3 = SVR(degree=3)
        svm4 = SVR(degree=4)
        svm5 = SVR(degree=5)

        svm1.fit(X_train, y_train)
        svm2.fit(X_train, y_train)
        svm3.fit(X_train, y_train)
        svm4.fit(X_train, y_train)
        svm5.fit(X_train, y_train)

        estimators = [('SVM1 : ', svm1), ('SVM2 : ', svm2), ('SVM3 : ', svm3), ('SVM4 : ', svm4), ('SVM5 : ', svm5)]

        for estimator in estimators:
            acc = cross_val_score(estimator[1], X_train, y_train, cv=10, scoring='r2')
            st.sidebar.text(f'{estimator[0]}{np.round(np.mean(acc), 2)}')

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf = VotingRegressor(estimators=estimators)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    XX = np.arange(start = X.min()-1 , stop = X.max()+1 , step = 0.01).reshape(-1,1)
    input_array = clf.predict(XX).reshape(-1,1)

    ax.plot(XX, input_array , 'r' , lw = 1.5)
    #ax.xlabel("Col1")
    #ax.ylabel("Col2")
    orig = st.pyplot(fig)
    if '5-SVM' in models:
        st.subheader("r2 for Voting Ensemble Regresser  " + str(np.round(np.mean(cross_val_score(clf, X_train, y_train, cv=10, scoring='r2')), 2)))
    else:
        st.subheader("r2 for Voting Ensemble Regresser  " + str(round(accuracy_score(y_test, y_pred), 2)))

    figs.set_figwidth(cols)
    figs.set_figheight(1)
    ind = 0
    flag = 0
    while ind < len(models):
        i = models[ind]
        if 'Linear Regression' == i:
            axs[ind - flag].scatter(X, y, alpha=0.5, s=1)
            axs[ind-flag].plot(X_test, lr_pred , 'r--' , lw = 0.3)
            axs[ind-flag].set_axis_off()
            axs[ind-flag].set_title('Linear Regression' , fontsize=5)
        elif 'SVR' == i:
            axs[ind - flag].scatter(X, y, alpha=0.5, s=1)
            axs[ind - flag].plot(XX, svc.predict(input_array).reshape(XX.shape), 'r--' , lw = 0.3)
            axs[ind - flag].set_axis_off()
            axs[ind - flag].set_title('SVR', fontsize=5)
        elif 'Decision Tree Regresser' == i:
            axs[ind - flag].scatter(X, y, alpha=0.5, s=1)
            axs[ind - flag].plot(XX, rf.predict(input_array).reshape(XX.shape), 'r--' , lw = 0.3)
            axs[ind - flag].set_axis_off()
            axs[ind - flag].set_title('Decision Tree Regresser', fontsize=5)
        elif 'KNN'== i:
            axs[ind - flag].scatter(X, y, alpha=0.5, s=1)
            axs[ind - flag].plot(XX, knn.predict(input_array).reshape(XX.shape), 'r--' , lw = 0.3)
            axs[ind - flag].set_axis_off()
            axs[ind - flag].set_title('KNN', fontsize=5)
        elif '5-SVM' == i:
            flag = 1
        ind+=1

    ind -= 2
    if flag == 1:
        axs[ind+1].scatter(X, y, alpha=0.5, s=1)
        axs[ind+1].plot(XX, svm1.predict(input_array).reshape(XX.shape), 'r--' , lw = 0.3)
        axs[ind+1].set_axis_off()
        axs[ind+1].set_title('Degree-1', fontsize=5)

        axs[ind+2].scatter(X, y, alpha=0.5, s=1)
        axs[ind+2].plot(XX, svm2.predict(input_array).reshape(XX.shape), 'r--' , lw = 0.3)
        axs[ind+2].set_axis_off()
        axs[ind+2].set_title('Degree-2', fontsize=5)

        axs[ind+3].scatter(X, y, alpha=0.5, s=1)
        axs[ind+3].plot(XX, svm3.predict(input_array).reshape(XX.shape), 'r--' , lw = 0.3)
        axs[ind+3].set_axis_off()
        axs[ind+3].set_title('Degree-3', fontsize=5)

        axs[ind+4].scatter(X, y, alpha=0.5, s=1)
        axs[ind+4].plot(XX, svm4.predict(input_array).reshape(XX.shape), 'r--' , lw = 0.3)
        axs[ind+4].set_axis_off()
        axs[ind+4].set_title('Degree-4', fontsize=5)

        axs[ind+5].scatter(X, y, alpha=0.5, s=1)
        axs[ind+5].plot(XX, svm5.predict(input_array).reshape(XX.shape), 'r--' , lw = 0.3)
        axs[ind+5].set_axis_off()
        axs[ind+5].set_title('Degree-5', fontsize=5)

    st.pyplot(figs)




# open in pycharm or vs code
# after open terminal in that
# write command
# streamlit run voting-ensemble-regresser-viz.py
