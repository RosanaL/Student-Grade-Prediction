import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
file_path = '/mnt/student_analysis/dataset.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 按照条件分割数据
df_pos = df[df['Above_70'] == 1]  # Above_70 = 1 的样本
df_neg = df[df['Above_70'] == 0]  # Above_70 = 0 的样本

# 从负类中随机选择160个样本
df_neg_sampled = df_neg.sample(n=160, random_state=42)
df_pos_sampled = df_pos.sample(n=160, random_state=42)

# 合并正类样本和负类样本
df_balanced = pd.concat([df_pos_sampled, df_neg_sampled])

# 特征和目标变量
features = ['Questions_Authored', 'Answers_Submitted', 'Answers_Correct', 'Comments_Written', 'Total_Character_Count',
            'Student_Avg_Score']
X = df_balanced[features]
y = df_balanced['Above_70']

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据归一化处理
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化 GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 训练模型
model.fit(X_train_scaled, y_train)

# 预测
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

# 计算评价指标
mse = mean_squared_error(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# 精度、召回率、特异性
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')

# 判断是否满足性能要求
recall_threshold = 0.6
specificity_threshold = 0.6
precision_threshold = 0.6

if recall >= recall_threshold and specificity >= specificity_threshold and precision >= precision_threshold:
    # 输出数据到新的xlsx文件
    output_file_path = '/mnt/student_analysis/Gradient_balanced_dataset.xlsx'
    df_balanced.to_excel(output_file_path, index=False)
    print(f"Balanced dataset saved to {output_file_path}")

    # 可视化混淆矩阵并保存到文件
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('/mnt/student_analysis/Gradient_confusion_matrix.png')
    plt.close()
else:
    print("Performance criteria not met.")
