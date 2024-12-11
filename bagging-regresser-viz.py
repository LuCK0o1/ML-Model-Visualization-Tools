import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score as accuracy_score
from sklearn.ensemble import BaggingRegressor


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

    ax[0].scatter(X , y, c='b' , edgecolor = 'black')
    ax[0].set_axis_off()
    ax[0].set_title(model, fontsize=10)

    ax[1].scatter(X , y, c='b' , edgecolor = 'black')
    ax[1].set_axis_off()
    ax[1].set_title('Bagging Regresser', fontsize=10)

    return X, y

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Bagging Regressor")

data = st.sidebar.selectbox(
    'Data',
    ('Line','S-Shape','Sharp')
)

model = st.sidebar.selectbox(
    'Models',
    ('Decision Tree Regresser', 'SVR', 'KNN')
)

n_estimators = int(st.sidebar.slider('n_estimators : (model count)' , min_value=1 , max_value=500))

max_samples = float(st.sidebar.slider('max_sample : radnom sample row' , min_value=0.01 , max_value= 1.0))

bootstrap = st.sidebar.selectbox(
    'bootstrap',
    (True , False)
)

max_features = float(st.sidebar.slider('max_features : radnom sample col' , min_value=0.01 , max_value= 1.0))

bootstrap_features = st.sidebar.selectbox(
    'bootstrap_features',
    (True , False)
)

oob_score = st.sidebar.selectbox(
    'oob_score : Out Of Box',
    (True , False)
)

# Load initial graph
fig, ax = plt.subplots(1, 2)
fig.set_figheight(3)

# Plot initial graph
X,y = load_initial_graph(data = data , ax = ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if model == 'Decision Tree Regresser':
    model1 = DecisionTreeRegressor()
    clf1 = DecisionTreeRegressor()
    clf1.fit(X_train , y_train)
    pred = clf1.predict(X_test)
    acc = accuracy_score(y_test, pred)

elif model == 'SVR':
    model1 = SVR()
    clf1 = SVR()
    clf1.fit(X_train , y_train)
    pred = clf1.predict(X_test)
    acc = accuracy_score(y_test, pred)

else:
    model1 = KNeighborsRegressor()
    clf1 = KNeighborsRegressor()
    clf1.fit(X_train , y_train)
    pred = clf1.predict(X_test)
    acc = accuracy_score(y_test, pred)

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf = BaggingRegressor(estimator=model1,          # algo
                            n_estimators=n_estimators,         # int
                            max_samples=max_samples,         # int or float
                            max_features=max_features,        # int or float
                            bootstrap=bootstrap,          # bool
                            bootstrap_features=bootstrap_features,# bool
                            oob_score=oob_score,         # bool
                            warm_start=False,        # bool
                            n_jobs=None,             # int
                            random_state=None,       # int
                            verbose=0)               # int
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    input_array = np.arange(start = X.min()-1 , stop = X.max()+1 , step = 0.01).reshape(-1,1)
    labels = clf.predict(input_array).reshape(-1,1)

    fig.set_figheight(3)
    #fig.set_figwidth()
    # single model
    ax[0].plot(input_array, clf1.predict(input_array).reshape(input_array.shape), 'r' , lw = 1.5)
    ax[0].set_axis_off()
    ax[0].set_title(model , fontsize=10)
    ax[0].annotate(model + f' r2 Score: {np.round(acc,3)}',
                xy=(1.0, -0.2),
                xycoords='axes fraction',
                ha='right',
                va="center",
                fontsize=10)

    # bagging
    ax[1].plot(input_array, labels.reshape(input_array.shape), 'r' , lw = 1.5)
    ax[1].set_axis_off()
    ax[1].set_title('Bagging Regresser', fontsize=10)
    ax[1].annotate(f'Bagging Regresser r2 Score : {round(accuracy_score(y_test, y_pred), 3)}',
                   xy=(1.0, -0.2),
                   xycoords='axes fraction',
                   ha='right',
                   va="center",
                   fontsize=10)
    orig = st.pyplot(fig)

st.sidebar.text('[Bagging : bootstrap = True , bootstrap_features = False]\n\n'+
                '[Pasting : bootstrap = False , bootstrap_features = False]\n\n'+
                '[Random Subspace : boostrap = False , bootstrap_features = True]\n\n'+
                '[Random Patches : bootstrap = True , bootstrap_features = True]')

if bootstrap and bootstrap_features:
    type = 'Random Patches'
elif bootstrap and not bootstrap_features:
    type = 'Bagging'
elif not bootstrap and not bootstrap_features:
    type = 'Pasting'
else:
    type = 'Random Subspace'

st.subheader(type)

# open in pycharm or vs code
# after open terminal in that
# write command
# streamlit run bagging-regresser-viz.py