import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification , make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

def load_initial_graph(ax):
    X, y = make_classification(n_samples=100, 
                           n_features=2, 
                           n_informative=2, 
                           n_redundant=0, 
                           random_state=5,
                           class_sep=0.5)
    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
    return X,y

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array


plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Decision Tree Classifier")

criterion = st.sidebar.selectbox(
    'Criterion',
    ("gini", "entropy", "log_loss")
)

splitter = st.sidebar.selectbox(
    'Splitter',
    ("best", "random")
)

max_depth = st.sidebar.selectbox(
    'max_depth',
    [None] + list(range(1,301))
)

min_samples_split = int(st.sidebar.slider('min_samples_split',min_value=2 , max_value=375)) # int , float

min_samples_leaf = int(st.sidebar.slider('min_samples_leaf',min_value=1 , max_value=375)) # int , float

min_weight_fraction_leaf = float(st.sidebar.slider('min_weight_fraction_leaf',min_value=0.0 , max_value=0.5)) # float

max_features = st.sidebar.selectbox(
    'max_features',
    [None , 1, 2]
)

random_state = st.sidebar.selectbox(
    'random_state',
    [None] + list(range(1,301))
)

max_leaf_nodes= st.sidebar.selectbox(
    'max_leaf_nodes',
    [None] + list(range(2,301))
)

min_impurity_decrease= float(st.sidebar.number_input('min_impurity_decrease',value=0.0)) # float >= 0

# Load initial graph
fig, ax = plt.subplots()
fig1 , ax1 = plt.subplots()

# Plot initial graph
X,y = load_initial_graph(ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf = DecisionTreeClassifier(criterion=criterion, # {"gini", "entropy", "log_loss"}
                                splitter=splitter, # {"best", "random"}
                                max_depth=max_depth, # int
                                min_samples_split=min_samples_split, # int , float
                                min_samples_leaf=min_samples_leaf, # int , float
                                min_weight_fraction_leaf=min_weight_fraction_leaf, # float
                                max_features=max_features, # int, float or {"sqrt", "log2"}
                                random_state=random_state, # int
                                max_leaf_nodes=max_leaf_nodes, # int
                                min_impurity_decrease=min_impurity_decrease, # float >= 0
                                class_weight=None, # int
                                ccp_alpha=0.0, # int
                                monotonic_cst=None) # array[int]
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)
    st.subheader("Accuracy for Decision Tree  " + str(round(accuracy_score(y_test, y_pred), 2)))

    fig1.set_figheight(25)
    fig1.set_figwidth(20)
    ax1 = tree.plot_tree( clf,
                          feature_names=['f1','f2'],
                          class_names=['False','True'],
                          filled=True)
    st.pyplot(fig1)

# open in pycharm or vs code
# after open terminal in that
# write command
# streamlit run dt-viz.py
