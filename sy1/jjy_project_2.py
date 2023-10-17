from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import tqdm
from sklearn.metrics import precision_recall_fscore_support


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 40 * 30, 64)  # 减少全连接层的节点数
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 4)  # 输出层，假设有4个类别

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


def Train(input_path):
    # 定义数据集划分比例
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # 1. 准备数据集
    transform = transforms.Compose([transforms.Resize((60, 80)),
                                    transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图像
                                    transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=input_path, transform=transform)

    # 计算数据集划分的大小
    total_size = len(train_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # 使用random_split函数切分数据集
    train_subset, val_subset, test_subset = random_split(train_dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    train(train_loader, test_loader, test_subset)


def train(train_loader, test_loader, test_dataset):
    # 3. 定义损失函数和优化器
    model = Model().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 创建TensorBoard SummaryWriter
    writer = SummaryWriter()

    # 5. 训练模型
    epochs = 10
    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.to('cpu').item()

            # 记录训练损失到TensorBoard，使用step参数来区分不同的训练步骤
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + step)
            # 在每个训练步结束后输出预测结果
            eval(model, test_loader, test_dataset, writer, epoch * len(train_loader) + step)

    # 关闭TensorBoard SummaryWriter
    writer.close()

    # 保存模型参数
    torch.save(model.state_dict(), 'my_model_params.pth')


def eval(model, test_loader, test_dataset, writer, epoch):
    model.eval()  # 切换到评估模式
    y_true = []  # 存储真实标签
    y_pred = []  # 存储模型预测的标签
    num_samples = 25  # 要显示的样本数量

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # 预测的类别
            y_true += labels.cpu().tolist()
            y_pred += predicted.cpu().tolist()

        # 计算precision、recall、F1-score
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        # print(f"Epoch {epoch} - Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")

        # 记录评估指标到TensorBoard
        writer.add_scalar('Precision', precision, epoch)
        writer.add_scalar('Recall', recall, epoch)
        writer.add_scalar('F1-score', f1_score, epoch)

        # # 随机选择一些测试样本
        # sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        #
        # # 创建一个包含5x5子图的图像
        # num_rows = 5
        # num_cols = 5
        # fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 7.5))
        #
        # for i, index in enumerate(sample_indices):
        #     sample_image, sample_label = test_dataset[index]
        #     sample_image = sample_image.unsqueeze(0)  # 添加批次维度
        #
        #     # 绘制原始图像
        #     axes[i // 5, i % 5].imshow(sample_image.squeeze().cpu().numpy(), cmap='gray')
        #     axes[i // 5, i % 5].set_title(f'T: {sample_label}, P: {y_pred[index]}')
        #     axes[i // 5, i % 5].axis('off')
        #
        # # 调整子图之间的间距
        # plt.subplots_adjust(wspace=0.3, hspace=0.2)
        #
        # # 保存图像到TensorBoard
        # writer.add_figure('Sample Predictions', fig, global_step=epoch)


def save_model():
    # 导出模型为ONNX格式
    dummy_input = torch.randn(1, 1, 60, 80)  # 输入数据的示例
    print("saved")
    torch.onnx.export(Model(), dummy_input, "model.onnx", verbose=True)


if __name__ == "__main__":
    input_folder = "MDSD_subset_plus"
    output_folder = "redataset_2"
    Train(output_folder)
    Train(input_folder)
    # save_model()
