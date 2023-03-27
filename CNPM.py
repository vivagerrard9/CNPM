import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
import csv


import warnings
warnings.filterwarnings('ignore')
data = 'D:\Documents\Code\Book1.csv'
df = pd.read_csv(data, header=None, sep=',')

col_names =['Tuổi', 'Giới tính', 'Ô nhiễm không khí', 'Sử dụng rượu', 'Dị ứng bụi', 'Nguy cơ nghề nghiệp', 'Yếu tố di truyền', 'Bệnh phổi mãn tính', 'Chế độ ăn',
            'Béo phì', 'Hút thuốc', 'Ảnh hưởng khói thuốc', 'Sơn móng tay', 'Môi trường lạnh', 'Chuẩn đoán'    ]
df.columns = col_names

#print (df)
x = [var for var in df.columns if df[var].dtype!='O']
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
x_train, x_test, y_train, y_test = train_test_split(df[x], df['Chuẩn đoán'], test_size = 0.3, random_state = 0)

#print(x_test)
#print(y_test)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
pred = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score
#print('Model accuracy score: {0:0.2f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = clf.predict(x_train)
#print('Training-set accuracy score: {0:0.2f}'. format(accuracy_score(y_train, y_pred_train)))
#print('Training set score: {:.2f}'.format(gnb.score(x_train, y_train)))
#print('Test set score: {:.2f}'.format(gnb.score(x_test, y_test)))
null_acc = (31/(31+23))
#print('Null accuracy score: {:.2f}'.format(null_acc))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print('Confusion matrix\n\n', cm)
#print('\nTrue Positives(TP) = ', cm[0,0])
#print('\nTrue Negatives(TN) = ', cm[1,1])
#print('\nFalse Positives(FP) = ', cm[0,1])
#print('\nFalse Negatives(FN) = ', cm[1,0])

#TP = cm[0,0]
#TN = cm[1,1]
#FP = cm[0,1]
#FN = cm[1,0]


name = input('Mời bạn nhập họ và tên: ')
a = input('Tuổi: ')
print ('------------------------------------------------------------')
print ('\nBạn hãy trả lời câu hỏi bằng stt câu trả lời (1,2,3,...)\n')
print ('------------------------------------------------------------')
b = input('Giới tính: \n1.Nam \n2.Nữ \n')
c = input('Chỗ bạn ở có ô nhiễm không khí không? \n1.Thấp \n2.Trung bình \n3.Cao \n4.Rất cao \n')
d = input('Bạn có sử dụng rượu, bia không? \n1.Rất ít \n2.Ít \n3.Trung bình \n4.Nhiều \n5.Rất nhiều \n')
e = input('Bạn thường xuyên hít phải khói bụi không? \n1.Rất ít \n2.Ít \n3.Trung bình \n4.Nhiều \n5.Rất nhiều \n')
f = input('Nghề nghiệp của bạn có tiếp xúc với khói bụi hay không? \n1.Rất ít \n2.Ít \n3.Trung bình \n4.Nhiều \n5.Rất nhiều \n')
g = input('Gia đình bạn có ai mắc ung thư không? \n1.Thấp(không ai mắc) \n2.Trung bình(1 người mắc) \n3.Cao(từ 2 người trở lên) \n')
h = input('Bạn có bị bệnh phổi mãn tính không? \n1.Không \n2.Mắc nhưng đã chữa được \n3.Đang mắc \n4.Rất nặng \n ')
i = input('Chế độ ăn có đầy đủ chất dinh dưỡng không? \n1.Rất thấp \n2.Thấp \n3.Trung bình \n4.Cao \n5.Rất cao \n')
k = input('Bạn có bị béo phì không? \n1.Rất thấp(40-50kg) \n2.Thấp(50-60kg) \n3.Trung bình(60-70kg) \n4.Cao(70-80kg) \n5.Rất cao(trên 80kg) \n')
l = input('Bạn có hút thuốc không? \n1.Rất ít \n2.Ít \n3.Trung bình \n4.Nhiều \n5.Rất nhiều \n')
m = input('Bạn có thường xuyên ngửi khói thuốc không? \n1.Rất ít \n2.Ít \n3.Trung bình \n4.Nhiều \n5.Rất nhiều \n')
n = input('Bạn có sơn móng tay, móng chân không? \n1.Rất ít \n2.Ít \n3.Trung bình \n4.Nhiều \n5.Rất nhiều \n')
o = input('Bạn có sống ở môi trường lạnh không? \n1.Rất ít \n2.Ít \n3.Trung bình \n4.Nhiều \n5.Rất nhiều \n')

pre = clf.predict([[int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h),int(i),int(k),int(l),int(m),int(n),int(o)]])
if pre == [0]:
    print ('Bệnh nhân ' + name + ' khả năng cao mắc bệnh ung thư')
else:
    print ('Bệnh nhân ' + name + ' không mắc bệnh ung thư')

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication

Form, Window = uic.loadUiType("test.ui")

app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)
window.show()
app.exec()