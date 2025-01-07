import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix
import numpy as np

# 读取数据
file_path = '/mnt/student_analysis/balanced_dataset.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 特征和目标变量
features = ['Questions_Authored', 'Answers_Submitted', 'Answers_Correct', 'Comments_Written', 'Total_Character_Count',
            'Student_Avg_Score']
X = df[features]
y = df['Above_70']

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# 精度、召回率、特异性
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Specificity: {specificity:.4f}')


