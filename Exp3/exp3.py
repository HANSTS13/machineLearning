import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import(
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix;
roc_curve,
auc
)

data= load_breast_cancer()
x=data.data
y=data.target
print("Dataset Shape:",x.shape)
print("Classes:",np.unique(y))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify =y)
