import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn import tree

x1 = np.random.randn(1, 30)
y1 = 2 * x1 + 5 + np.random.rand(1, 30) * 1.4

x2 = 4 + np.random.randn(1, 30)
y2 = -0.3 * x2 + 12 + np.random.rand(1, 30) * 1.4

x3 = x1 + 8
y3 = -2 * x3 + 20 + np.random.rand(1, 30) * 1.4

X = np.concatenate([x1, x2, x3], axis=1)
y = np.concatenate([y1, y2, y3], axis=1)

def load_initial_graph(ax):
    ax.scatter(X, y, color='r')
    df = pd.DataFrame({0: X[0], 1: y[0]})
    return df

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Regression Tree Classifier")

criterion = st.sidebar.selectbox(
    'Criterion',
    ("squared_error", "friedman_mse", "absolute_error", "poisson")
)

splitter = st.sidebar.selectbox(
    'Splitter',
    ("best", "random")
)

max_depth = st.sidebar.selectbox(
    'max_depth',
    [None] + list(range(1,301))
)

min_samples_split = int(st.sidebar.slider('min_samples_split', min_value=2, max_value=375))  # int , float

min_samples_leaf = int(st.sidebar.slider('min_samples_leaf', min_value=1, max_value=375))  # int , float

min_weight_fraction_leaf = float(st.sidebar.slider('min_weight_fraction_leaf', min_value=0.0, max_value=0.5)) # float

max_features = st.sidebar.selectbox(
    'max_features',
    [None, 1]
)

random_state = st.sidebar.selectbox(
    'random_state',
    [None] + list(range(1, 301))
)

max_leaf_nodes = st.sidebar.selectbox(
    'max_leaf_nodes',
    [None] + list(range(2, 301))
)

min_impurity_decrease = float(st.sidebar.number_input('min_impurity_decrease', value=0.0))  # float >= 0

# Load initial graph
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()

# Plot initial graph
df = load_initial_graph(ax)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:1], df.iloc[:, 1:0:-1], test_size=0.2, random_state=3)
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf = DecisionTreeRegressor(criterion=criterion,  # {"squared_error", "friedman_mse", "absolute_error", "poisson"}
                                splitter=splitter,  # {"best", "random"}
                                max_depth=max_depth,  # int
                                min_samples_split=min_samples_split,  # int , float
                                min_samples_leaf=min_samples_leaf,  # int , float
                                min_weight_fraction_leaf=min_weight_fraction_leaf,  # float
                                max_features=max_features,  # int, float or {"sqrt", "log2"}
                                random_state=random_state,  # int
                                max_leaf_nodes=max_leaf_nodes,  # int
                                min_impurity_decrease=min_impurity_decrease,  # float >= 0
                                ccp_alpha=0.0,  # int
                                monotonic_cst=None)  # array[int]
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)


    XX = np.arange(start=min(df.iloc[:, 0])-1, stop=max(df.iloc[:, 1])+1, step=0.01).reshape(-1,1)
    YY = clf.predict(XX)
    #ax.linspace((-3, 12))
    ax.plot(XX, YY, color='b')

    plt.xlabel("input")
    plt.ylabel("output")
    orig = st.pyplot(fig)
    st.subheader("Accuracy for Regression Tree  " + str(round(r2_score(y_test, y_pred), 2)))

    fig1.set_figheight(25)
    fig1.set_figwidth(20)
    ax1 = tree.plot_tree(clf,
                         feature_names=['xi'],
                         class_names=['False', 'True'],
                         filled=True)
    st.pyplot(fig1)

# open in pycharm or vs code
# after open terminal in that
# write command
# streamlit run dt-reg-viz.py