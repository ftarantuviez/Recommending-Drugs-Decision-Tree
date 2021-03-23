import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

from  io import StringIO
import pydotplus

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Recommending Drugs - Decision Trees', page_icon="./f.png")
st.title('Recommending Drugs - Decision Trees')
st.subheader('By [Francisco Tarantuviez](https://www.linkedin.com/in/francisco-tarantuviez-54a2881ab/) -- [Other Projects](https://franciscot.dev/portfolio)')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('---')
st.write("""
The idea of this app is use Decision Tree algorithm to build a model from historical data of patients, and their response to different medications. Then use the trained decision tree to predict the class of a unknown patient, or to find a proper drug for a new patient.


## About the dataset

Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and y. 

Part of the job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The feature sets of this dataset are Age, Sex, Blood Pressure, and Cholesterol of patients, and the target is the drug that each patient responded to.

It is a sample of multiclass classifier, and you can use the training part of the dataset to build a decision tree, and then use it to predict the class of a unknown patient, or to prescribe it to a new patient.

""")

@st.cache
def load_data():
  return pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv")
my_data = load_data()
st.dataframe(my_data)

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

y = my_data["Drug"]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

st.write(""" 
## Modeling

The model we are gonna build is depicted below. You can change the parameters from the left sidebar
""")
st.code("drugTree = DecisionTreeClassifier(criterion='entropy', max_depth = 4)")
st.sidebar.header("Customize the model")

st.sidebar.write("### Decision Tree Parameters")
criterion = st.sidebar.selectbox("Criterion", ["entropy","gini"])
splitter = st.sidebar.selectbox("Splitter", ["best", "random"]) 
max_depth = st.sidebar.slider("Max depth", 2, 8, 4)
drugTree = DecisionTreeClassifier(criterion=criterion, max_depth = max_depth, splitter=splitter)
drugTree.fit(X_trainset,y_trainset)
predTree = drugTree.predict(X_testset)

st.write(""" 
## Evaluation

Now we can see some metrics about our model
""")
st.dataframe(pd.DataFrame(pd.Series(metrics.accuracy_score(y_testset, predTree)), columns=["Accuracy"]))

st.write("### Confusion matrix")
sns.heatmap(metrics.confusion_matrix(y_testset, predTree), annot=True)
plt.title("Confusion matrix for {}".format("Decision Tree"))
st.pyplot()

st.write("## Visualization")

st.write("""
If you wanna visualize the tree, you have to press the below button.

**[WARNING]**: this task can be a little heavy for your computer.
""")
if st.button("Show tree"):
  dot_data = StringIO()
  filename = "drugtree.png"
  featureNames = my_data.columns[0:5]
  targetNames = my_data["Drug"].unique().tolist()
  out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
  graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
  graph.write_png(filename)
  img = mpimg.imread(filename)
  plt.figure(figsize=(100, 200))
  plt.imshow(img,interpolation='nearest')
  st.pyplot()
# This app repository

st.write("""
## Predict you sample

Here you can pass to the model a unseen sample to predict, and it will recommend you the drug to use
""")

col1, col2 = st.beta_columns(2)
age = col1.slider("Age", 1, 90, 30)
sex = col2.selectbox("Sex", ["M", "F"])
BP = col1.selectbox("BP", ["LOW", "HIGH", "NORMAL"])
cholesterol = col2.selectbox("Cholesterol", ["HIGH", "NORMAL"])
na_to_k  = st.number_input("Na to K")

if st.button("Run"):
  sex_input = 0 if sex == "F" else 1
  BP_input = 1
  if BP == "HIGH":
    BP_input = 0
  elif BP == "NORMAL":
    BP_input = 2
  else:
    BP_input = 1
  cholesterol_input = 0 if cholesterol == "HIGH" else 1

  input_user = [[age, sex_input, BP_input, cholesterol_input, na_to_k]]
  predictions = drugTree.predict(input_user)
  pred_proba = drugTree.predict_proba(input_user)
  
  st.write("**Results**")
  st.dataframe(pd.DataFrame(pd.Series(predictions), columns=["Prediction"]))


st.write("""
## App repository

[Github](https://github.com/ftarantuviez/)TODO
""")
# / This app repository