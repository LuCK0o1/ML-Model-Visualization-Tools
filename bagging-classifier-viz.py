import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier


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

    ax[0].scatter(X[:,0].T, X[:,1].T, c=y, cmap='rainbow')
    ax[0].set_axis_off()
    ax[0].set_title(model, fontsize=10)

    ax[1].scatter(X[:, 0].T, X[:, 1].T, c=y, cmap='rainbow')
    ax[1].set_axis_off()
    ax[1].set_title('Bagging Classifier', fontsize=10)

    return X, y

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array


plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Bagging Classifier")

data = st.sidebar.selectbox(
    'Data',
    ('Circle','Blob','Moons','Guassiun')
)

model = st.sidebar.selectbox(
    'Models',
    ('Decision Tree Classifier', 'SVC', 'KNN')
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

if model == 'Decision Tree Classifier':
    model1 = DecisionTreeClassifier()
    clf1 = DecisionTreeClassifier()
    clf1.fit(X_train , y_train)
    pred = clf1.predict(X_test)
    acc = accuracy_score(y_test, pred)

elif model == 'SVC':
    model1 = SVC()
    clf1 = SVC()
    clf1.fit(X_train , y_train)
    pred = clf1.predict(X_test)
    acc = accuracy_score(y_test, pred)

else:
    model1 = KNeighborsClassifier()
    clf1 = KNeighborsClassifier()
    clf1.fit(X_train , y_train)
    pred = clf1.predict(X_test)
    acc = accuracy_score(y_test, pred)

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf = BaggingClassifier(estimator=model1,          # algo
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

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    fig.set_figheight(3)
    #fig.set_figwidth()
    # single model
    ax[0].contourf(XX, YY, clf1.predict(input_array).reshape(XX.shape), alpha=0.5, cmap='rainbow')
    ax[0].set_axis_off()
    ax[0].set_title(model , fontsize=10)
    ax[0].annotate(model + f' acuuracy : {acc}',
                xy=(1.0, -0.2),
                xycoords='axes fraction',
                ha='right',
                va="center",
                fontsize=10)

    # bagging
    ax[1].contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    ax[1].set_axis_off()
    ax[1].set_title('Bagging Classifier', fontsize=10)
    ax[1].annotate(f'Bagging Classifier acuuracy : {round(accuracy_score(y_test, y_pred), 2)}',
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
# streamlit run bagging-classifier-viz.py