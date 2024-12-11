import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_initial_graph(ax):
    X, y = make_classification(n_samples=100,
                               n_features=2,
                               n_informative=2,
                               n_redundant=0,
                               random_state=5,
                               class_sep=0.5)
    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow' , edgecolor = 'black')
    return X,y

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array


plt.style.use('fivethirtyeight')

st.sidebar.markdown("# K Nearest Neighbor Classifier")

n_neighbors = int(st.sidebar.slider('n_neighbors', min_value=1, max_value=80))

weights = st.sidebar.selectbox(
    'weights',
    ('uniform', 'distance')
)

algorithm = st.sidebar.selectbox(
    'algorithm',
    ('auto', 'ball_tree', 'kd_tree', 'brute')
)

leaf_size = int(st.sidebar.slider('leaf_size',min_value=1 , max_value=80))

p = int(st.sidebar.slider('p',min_value=1 , max_value=80))

n_jobs = st.sidebar.selectbox(
    'n_jobs',
    (None, -1)
)

metric = st.sidebar.selectbox(
    'metric',
    ('minkowski', 'cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean')
)

if metric == 'minkowski':
    metric_params = {'p': int(st.sidebar.slider('metric_params : p(scalar)', min_value=1 , max_value=80))}
else:
    metric_params = None

# Load initial graph
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()

# Plot initial graph
X,y = load_initial_graph(ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf = KNeighborsClassifier(n_neighbors=n_neighbors,       # int
                               weights=weights,               # {'uniform', 'distance'}
                               algorithm=algorithm,           # {'auto', 'ball_tree', 'kd_tree', 'brute'}
                               leaf_size=leaf_size,           # int
                               p=p,                           # int
                               metric=metric,                 # {'minkowski', 'cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean'}
                               metric_params=metric_params,   # parameterts of selected metric in dict format
                               n_jobs=n_jobs)                 # int
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)
    st.subheader("Accuracy for Decision Tree  " + str(round(accuracy_score(y_test, y_pred), 2)))

    ax1 = plt.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    st.pyplot(fig1)

# open in pycharm or vs code
# after open terminal in that
# write command
# streamlit run knn_classifier_viz.py
