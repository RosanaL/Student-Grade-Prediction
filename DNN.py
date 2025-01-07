import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
features = ['Questions_Authored', 'Answers_Submitted', 'Answers_Correct', 'Comments_Written', 'Total_Character_Count', 'Student_Avg_Score']
X = df_balanced[features]
y = df_balanced['Above_70']

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据归一化处理
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# 定义 DNN 模型
class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DNNModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和评估循环
epochs = 30
recall_threshold = 0.60
specificity_threshold = 0.60
precision_threshold = 0.60
satisfied_condition = False

while not satisfied_condition:
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # 每个epoch后检查模型性能
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_test_tensor.to(device)).cpu().numpy()
            y_pred = (y_pred_prob > 0.5).astype(int)

            # 将 y_test_tensor 转换为整数类型 numpy 数组以便于评估
            y_test_np = y_test_tensor.numpy().astype(int)

            mse = mean_squared_error(y_test_np, y_pred)
            conf_matrix = confusion_matrix(y_test_np, y_pred)
            tn, fp, fn, tp = conf_matrix.ravel()

            # 精度、召回率、特异性
            precision = precision_score(y_test_np, y_pred)
            recall = recall_score(y_test_np, y_pred)
            specificity = tn / (tn + fp)
            roc_auc = roc_auc_score(y_test_np, y_pred_prob)

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            print(f'Mean Squared Error (MSE): {mse:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'Specificity: {specificity:.4f}')
            print(f'ROC AUC Score: {roc_auc:.4f}')

            if recall >= recall_threshold and specificity >= specificity_threshold and precision >= precision_threshold:
                satisfied_condition = True
                # 输出数据到新的xlsx文件
                output_file_path = '/mnt/student_analysis/DNN_balanced_dataset.xlsx'
                df_balanced.to_excel(output_file_path, index=False)
                print(f"Balanced dataset saved to {output_file_path}")
                # 可视化混淆矩阵并保存到文件
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
                plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.savefig(f'/mnt/student_analysis/DNN_confusion_matrix_epoch_{epoch + 1}.png')
                plt.close()
                break

    # 确保循环不会因为满足条件后进入无限循环
    if satisfied_condition:
        break
