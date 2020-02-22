import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('data.csv')
#malignant means canercous tissue
#benign means non cancerous tissue
#so here we have to find whether the tissue is cancerous or not
x=data.iloc[:,2:32].values
y=data.iloc[:,1].values
data.isnull().sum()

#label encoding because machine understands only the numerical values and not strings or characters
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
explained_variance=pca.explained_variance_ratio_

from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=1)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

