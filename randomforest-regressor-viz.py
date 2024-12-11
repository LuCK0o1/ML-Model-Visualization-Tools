import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import dtreeviz

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import  DecisionTreeRegressor , plot_tree


def load_initial_graph(data , ax):
    if data == 'Line':
        csv = Path("C:/Users/SHUBHAM/PycharmProjects/ML practice/Data/regg1.csv")
        df = pd.read_csv(csv.resolve(), index_col=0)
        X = np.array(df.iloc[:, 0]).reshape(-1, 1)
        y = np.array(df.iloc[:, 1]).reshape(-1, 1)
    elif data == 'S-Shape':
        csv = Path("C:/Users/SHUBHAM/PycharmProjects/ML practice/Data/regg2.csv")
        df = pd.read_csv(csv.resolve(), index_col=0)
        X = np.array(df.iloc[:, 0]).reshape(-1, 1)
        y = np.array(df.iloc[:, 1]).reshape(-1, 1)
    elif data == 'Sharp':
        csv = Path("C:/Users/SHUBHAM/PycharmProjects/ML practice/Data/regg3.csv")
        df = pd.read_csv(csv.resolve(), index_col=0)
        X = np.array(df.iloc[:, 0]).reshape(-1, 1)
        y = np.array(df.iloc[:, 1]).reshape(-1, 1)

    ax[0].scatter(X, y, c='b', edgecolor='black')
    ax[0].set_axis_off()
    ax[0].set_title('Decision Tree Regressor', fontsize=10)

    ax[1].scatter(X, y, c='b', edgecolor='black')
    ax[1].set_axis_off()
    ax[1].set_title('Random Forest Regresser', fontsize=10)

    return X, y

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Random Forest Regressor")

data = st.sidebar.selectbox(
    'Data',
    ('Line','S-Shape','Sharp')
)

n_estimators = int(st.sidebar.slider('n_estimators : (model count)' , min_value=1 , max_value=500))

criterion = st.sidebar.selectbox(
    'Criterion',
    ("squared_error", "absolute_error", "friedman_mse", "poisson")
)

max_depth = st.sidebar.selectbox(
    'max_depth',
    [None] + list(range(1,301))
)

min_samples_split = int(st.sidebar.slider('min_samples_split',min_value=2 , max_value=375)) # int , float

min_samples_leaf = int(st.sidebar.slider('min_samples_leaf',min_value=1 , max_value=375)) # int , float

min_weight_fraction_leaf = float(st.sidebar.slider('min_weight_fraction_leaf',min_value=0.0 , max_value=0.5)) # float

max_samples = float(st.sidebar.slider('max_sample (0.0 = None): radnom sample row' , min_value=0.0 , max_value= 1.0))

if max_samples == 0.0:
    max_samples = None

max_features = st.sidebar.selectbox(
    'max features',
    ("sqrt", "log2", None , 'manual')
)

if max_features == 'manual':
    manual = int(st.sidebar.slider('Type Feature Number', min_value=1 , max_value=2))
else:
    manual = None

max_leaf_nodes= st.sidebar.selectbox(
    'max_leaf_nodes',
    [None] + list(range(2,301))
)

bootstrap = st.sidebar.selectbox(
    'bootstrap',
    (True , False)
)

oob_score = st.sidebar.selectbox(
    'oob_score : Out Of Box',
    (True , False)
)

viz_tree = st.selectbox(
    f'Tree Visualizer : Total Trees (0 to {n_estimators - 1})',
    list(range(0, n_estimators))
)

# Load initial graph
fig, ax = plt.subplots(1, 2)
fig.set_figheight(3)

fig1 , ax1 = plt.subplots()

# Plot initial graph
X,y = load_initial_graph(data = data , ax = ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

clf1 = DecisionTreeRegressor()
clf1.fit(X_train , y_train)
pred = clf1.predict(X_test)
acc = accuracy_score(y_test, pred)

model = 'Decision Tree Regressor'

# Random Forest Model #

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf = RandomForestRegressor(n_estimators=n_estimators,  # int
                                 criterion=criterion,  # {"squared_error", "absolute_error", "friedman_mse", "poisson"}
                                 max_depth=max_depth,  # int
                                 min_samples_split=min_samples_split,  # int or float[0,1]
                                 min_samples_leaf=min_samples_leaf,  # int or float[0,1]
                                 min_weight_fraction_leaf=min_weight_fraction_leaf,  # float
                                 max_features=max_features if not manual else manual ,  # {"sqrt", "log2", None}
                                 max_leaf_nodes=max_leaf_nodes,  # int
                                 min_impurity_decrease=0.0,  # float
                                 bootstrap=bootstrap,  # bool
                                 oob_score=oob_score,  # bool or callable/custom
                                 n_jobs=None,  # None / -1
                                 random_state=None,  # int
                                 verbose=0,  # int
                                 warm_start=False,  # bool
                                 ccp_alpha=0.0,  # non nagative float
                                 max_samples=max_samples,  # int or float
                                 monotonic_cst=None)  # array(int)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    input_array = np.arange(start=X.min() - 1, stop=X.max() + 1, step=0.01).reshape(-1, 1)
    labels = clf.predict(input_array).reshape(-1, 1)

    fig.set_figheight(3)
    # fig.set_figwidth()
    # single model
    ax[0].plot(input_array, clf1.predict(input_array).reshape(input_array.shape), 'r', lw=1.5)
    ax[0].set_axis_off()
    ax[0].set_title(model, fontsize=10)
    ax[0].annotate(model + f' r2 Score: {np.round(acc, 3)}',
                   xy=(1.0, -0.2),
                   xycoords='axes fraction',
                   ha='right',
                   va="center",
                   fontsize=10)

    # bagging
    ax[1].plot(input_array, labels.reshape(input_array.shape), 'r', lw=1.5)
    ax[1].set_axis_off()
    ax[1].set_title('Random Forest Regressor', fontsize=10)
    ax[1].annotate(f'Random Forest Regressor r2 Score : {round(accuracy_score(y_test, y_pred), 3)}',
                   xy=(1.0, -0.2),
                   xycoords='axes fraction',
                   ha='right',
                   va="center",
                   fontsize=10)
    orig = st.pyplot(fig)

    # Tree Viz

    fig1.set_figheight(25)
    fig1.set_figwidth(20)
    ax1 = plot_tree(clf.estimators_[viz_tree],
                    feature_names=['f1', 'f2'],
                    filled=True)
    st.pyplot(fig1)


# open in pycharm or vs code
# after open terminal in that
# write command
# streamlit run randomforest-regressor-viz.py