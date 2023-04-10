import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
import csv
import math


import warnings
warnings.filterwarnings('ignore')
data = 'D:\Documents\Code\Book1.csv'
df = pd.read_csv(data, header=None, sep=',')

col_names =['Tuổi', 'Giới tính', 'Ô nhiễm không khí', 'Sử dụng rượu', 'Dị ứng bụi', 'Nguy cơ nghề nghiệp', 'Yếu tố di truyền', 'Bệnh phổi mãn tính', 'Chế độ ăn',
            'Béo phì', 'Hút thuốc', 'Ảnh hưởng khói thuốc', 'Sơn móng tay', 'Môi trường lạnh', 'Chuẩn đoán'    ]
df.columns = col_names

x1 = df[['Ô nhiễm không khí', 'Dị ứng bụi', 'Nguy cơ nghề nghiệp', 'Yếu tố di truyền', 'Hút thuốc', 'Ảnh hưởng khói thuốc', 'Béo phì', 'Chế độ ăn', 'Sơn móng tay', 'Môi trường lạnh', 'Bệnh phổi mãn tính']]
x2 = df[['Ô nhiễm không khí', 'Dị ứng bụi', 'Nguy cơ nghề nghiệp', 'Yếu tố di truyền', 'Bệnh phổi mãn tính', 'Hút thuốc', 'Ảnh hưởng khói thuốc']]
x3 = df[['Chế độ ăn', 'Ảnh hưởng khói thuốc', 'Sử dụng rượu', 'Yếu tố di truyền', 'Dị ứng bụi', 'Nguy cơ nghề nghiệp', 'Béo phì']]
x4 = df[['Hút thuốc', 'Ô nhiễm không khí', 'Dị ứng bụi', 'Nguy cơ nghề nghiệp', 'Yếu tố di truyền', 'Bệnh phổi mãn tính', 'Ảnh hưởng khói thuốc']]

y = [var for var in df.columns if df[var].dtype=='O']
#preprocessing
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
yle = le.fit_transform(df[y])  
df['Chuẩn đoán'] = yle

#ckeck null
#print(df[x].isnull().sum())
#print('---------------------------------------------------')
#print(df[y].isnull().sum())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1, df['Chuẩn đoán'],test_size = 0.3, random_state = 0)

#print(x_test)
#print(y_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
pred = knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.2f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = knn.predict(x_train)
print('Training-set accuracy score: {0:0.2f}'. format(accuracy_score(y_train, y_pred_train)))
print('Training set score: {:.2f}'.format(knn.score(x_train, y_train)))
print('Test set score: {:.2f}'.format(knn.score(x_test, y_test)))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

acc = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
TPr = TP/(TP+FN) #Recall
TNr = TN/(TN+FP)
G = TPr*TNr
Gmean = math.sqrt(G)

print (Gmean)