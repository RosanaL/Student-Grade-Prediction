import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix, roc_auc_score

# 读取数据
file_path = '/mnt/student_analysis/balanced_dataset.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 特征和目标变量
features = ['Questions_Authored', 'Answers_Submitted', 'Answers_Correct', 'Comments_Written', 'Total_Character_Count', 'Student_Avg_Score']
X = df[features]
y = df['Above_70']

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据归一化处理
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化模型
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)

# 评估
mse = mean_squared_error(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# 精度、召回率、特异性
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')
