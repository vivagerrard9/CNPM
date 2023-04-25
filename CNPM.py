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
x = df[[ 'Ô nhiễm không khí', 'Sử dụng rượu', 'Dị ứng bụi', 'Nguy cơ nghề nghiệp', 'Yếu tố di truyền', 'Bệnh phổi mãn tính', 'Chế độ ăn',
            'Béo phì', 'Hút thuốc', 'Ảnh hưởng khói thuốc', 'Sơn móng tay', 'Môi trường lạnh']]

y = [var for var in df.columns if df[var].dtype=='O']
#preprocessing
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
yle = le.fit_transform(df[y])  
df['Chuẩn đoán'] = yle

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, df['Chuẩn đoán'], test_size = 0.3, random_state = 0)


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
pred = clf.fit(x, df['Chuẩn đoán'])


import tkinter as tk
from tkinter import ttk


questions = [
    {
        "question": "1.Chỗ bạn ở có ô nhiễm không khí không?",
        "options": ["1.Thấp", "2.Trung bình", "3.Cao", "4.Rất cao"]
    },
    {
        "question": "2.Bạn có sử dụng rượu, bia không?",
        "options": ["1.Rất ít", "2.Ít", "3.Trung bình", "4.Nhiều", "5.Rất nhiều"]
    },
    {
        "question": "3.Bạn thường xuyên hít phải khói bụi không?",
        "options": ["1.Rất ít", "2.Ít", "3.Trung bình", "4.Nhiều", "5.Rất nhiều"]
    },
    {
        "question": "4.Nghề nghiệp của bạn có tiếp xúc với khói bụi hay không?",
        "options": ["1.Rất ít", "2.Ít", "3.Trung bình", "4.Nhiều", "5.Rất nhiều"]
    },
    {
        "question": "5.Gia đình bạn có ai mắc ung thư không?",
        "options": ["1.Thấp(không ai mắc)", "2.Trung bình(1 người mắc)", "3.Cao(từ 2 người trở lên)"]
    },
    {
        "question": "6.Bạn có bị bệnh phổi mãn tính không?",
        "options": ["1.Không", "2.Mắc nhưng đã chữa được", "3.Đang mắc", "4.Rất nặng"]
    },
    {
        "question": "7.Chế độ ăn có đầy đủ chất dinh dưỡng không?",
        "options": ["1.Thấp", "2.Trung bình", "3.Cao", "4.Rất cao"]
    },
    {
        "question": "8.Bạn có bị béo phì không?",
        "options": ["1.Rất thấp(40-50kg)", "2.Thấp(50-60kg)", "3.Trung bình(60-70kg)", "4.Cao(70-80kg)", "5.Rất cao(trên 80kg)"]
    },
    {
        "question": "9.Bạn có hút thuốc không?",
        "options": ["1.Rất ít", "2.Ít", "3.Trung bình", "4.Nhiều", "5.Rất nhiều"]
    },
    {
        "question": "10.Bạn có thường xuyên ngửi khói thuốc không?",
        "options": ["1.Rất ít", "2.Ít", "3.Trung bình", "4.Nhiều", "5.Rất nhiều"]
    },
    {
        "question": "11.Bạn có sơn móng tay, móng chân không?",
        "options": ["1.Rất ít", "2.Ít", "3.Trung bình", "4.Nhiều", "5.Rất nhiều"]
    },
    {
        "question": "12.Bạn có sống ở môi trường lạnh không?",
        "options": ["1.Rất ít", "2.Ít", "3.Trung bình", "4.Nhiều", "5.Rất nhiều"]
    }
]


ten = None
tuoi = None

def hien_thi_cau_hoi():
    global ten, tuoi
    ten = ten_var.get()
    tuoi = tuoi_var.get()

    if not ten or not tuoi:
        answer_label.config(text="Vui lòng nhập đầy đủ thông tin.")
        return

    window.destroy()
    return ten, tuoi


window = tk.Tk()
window.title("Nhập tên và tuổi")
window.geometry("300x400")

ten_label = tk.Label(window, text="Họ và Tên:")
ten_label.pack()
ten_var = tk.StringVar()
ten_entry = tk.Entry(window, textvariable=ten_var)
ten_entry.pack()

tuoi_label = tk.Label(window, text="Tuổi:")
tuoi_label.pack()
tuoi_var = tk.IntVar()
tuoi_entry = tk.Entry(window, textvariable=tuoi_var)
tuoi_entry.pack()

ok_button = tk.Button(window, text="OK", command=hien_thi_cau_hoi)
ok_button.pack()
window.mainloop()

window_open = True
def submit_answer():
    global ten, tuoi
    if ten and tuoi:
        selected_answers = []
        for i in range(len(questions)):
            selected_option = var_list[i].get()
            selected_answers.append(selected_option)

        selected_answers = np.array(selected_answers).reshape(1, -1)
        if selected_answers.shape[1] != 12:
            answer_label.config(text="Vui lòng chọn đầy đủ đáp án.")
            return
        if answer_label.winfo_exists():
            pre = clf.predict(selected_answers)
            if pre == [0]:
                answer_label.config(text="Bệnh nhân " + ten + " dự đoán mắc bệnh")
            else:
                answer_label.config(text="Bệnh nhân " + ten + " dự đoán khỏe mạnh")
    else:
        answer_label.config(text="Vui lòng nhập đầy đủ thông tin.")
    
    

window = tk.Tk()
window.title("Câu hỏi và câu trả lời")
window.geometry("800x600")
canvas = tk.Canvas(window)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)

scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

content_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=content_frame, anchor=tk.NW)    

var_list = []


for i in range(len(questions)):
    question_label = tk.Label(content_frame, text=questions[i]["question"], justify="left")
    question_label.pack(pady=10)

    var = tk.IntVar()
    var_list.append(var)

    for j in range(len(questions[i]["options"])):
        option = tk.Radiobutton(content_frame, text=questions[i]["options"][j], variable=var, value=j + 1)
        option.pack()




submit_button = tk.Button(window, text="Submit", command=submit_answer)
submit_button.pack(pady=10)

canvas.bind_all('<MouseWheel>', lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))
canvas.configure(scrollregion=canvas.bbox('all'))

answer_label = tk.Label(window, text="")
answer_label.pack()

window.mainloop()

