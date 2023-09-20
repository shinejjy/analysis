import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def RGB2HSV(input_folder, output_folders):
    # 加入输出算子
    output_folders = [os.path.join("E:\\jjy\\class\\2023-2024 1\\数字音视频分析\\sy1\\redataset", folder) for folder in output_folders]
    # 创建输出文件夹（如果不存在）
    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)
    # 循环处理每个子文件夹中的图像
    for i, folder in enumerate(output_folders, start=1):
        input_subfolder = os.path.join(input_folder, f"class{i}")
        output_subfolder = folder

        # 遍历子文件夹中的所有图像文件
        for j, filename in enumerate(os.listdir(input_subfolder)):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # 可根据实际情况修改文件扩展名
                # 读取图像
                image_path = os.path.join(input_subfolder, filename)
                image = cv2.imread(image_path)

                # 转换为HSV色彩空间
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # 提取亮度（灰度）通道
                brightness_channel = hsv_image[:, :, 2]
                # 进行直方图均衡化
                equalized_image = cv2.equalizeHist(brightness_channel)

                # 构建输出文件路径
                output_path = os.path.join(output_subfolder, filename)
                # 保存亮度图像
                cv2.imwrite(output_path, equalized_image)
                
                if i==1 and j==0:
                    # 可视化过程
                    plt.figure(figsize=(10, 3))
                    
                    plt.subplot(131)
                    plt.title('Original Image')
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    plt.axis('off')

                    plt.subplot(132)
                    plt.title('Bright Image')
                    plt.imshow(brightness_channel, cmap='gray')
                    plt.axis('off')

                    plt.subplot(133)
                    plt.title('Equalized Brightness')
                    plt.imshow(equalized_image, cmap='gray')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.show()

                    # 可视化过程
                    plt.figure(figsize=(10/3*2, 3))
                    
                    plt.subplot(121)
                    plt.title('Brightness Histogram')
                    plt.hist(brightness_channel.flatten(), bins=256, range=(0, 256), density=True, color='red', alpha=0.6)

                    plt.subplot(122)
                    plt.title('Equalized Brightness Histogram')
                    plt.hist(equalized_image.flatten(), bins=256, range=(0, 256), density=True, color='red', alpha=0.6)

                    plt.tight_layout()
                    plt.show()

                    # 分别提取HSV通道
                    h_channel = hsv_image[:, :, 0]
                    s_channel = hsv_image[:, :, 1]
                    v_channel = hsv_image[:, :, 2]

                    # 显示原始RGB图像
                    plt.figure(figsize=(12, 4))
                    plt.subplot(221)
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    plt.title('Original RGB Image')
                    plt.axis('off')

                    # 显示HSV通道
                    plt.subplot(222)
                    plt.imshow(h_channel, cmap='hsv', vmin=0, vmax=179)
                    plt.title('Hue (H) Channel')
                    plt.axis('off')

                    plt.subplot(223)
                    plt.imshow(s_channel, cmap='gray', vmin=0, vmax=255)
                    plt.title('Saturation (S) Channel')
                    plt.axis('off')

                    plt.subplot(224)
                    plt.imshow(v_channel, cmap='gray', vmin=0, vmax=255)
                    plt.title('Value (V) Channel')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.show()

    print("处理完成！")


def visible_feature(X_train_lda, y_train, svm, surface_index):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取每个类别的数据
    for label in np.unique(y_train):
        ax.scatter(X_train_lda[y_train == label, 0], X_train_lda[y_train == label, 1], X_train_lda[y_train == label, 2],
                   label=f'Class {label}')

    ax.set_xlabel('LDA Component 1')
    ax.set_ylabel('LDA Component 2')
    ax.set_zlabel('LDA Component 3')

    # 绘制SVM的决策超平面
    xx, yy = np.meshgrid(np.linspace(X_train_lda[:, 0].min(), X_train_lda[:, 0].max(), 50),
                         np.linspace(X_train_lda[:, 1].min(), X_train_lda[:, 1].max(), 50))
    zz = (-svm.intercept_[surface_index] - svm.coef_[surface_index, 0] * xx - svm.coef_[surface_index, 1] * yy) / \
         svm.coef_[surface_index, 2]
    ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='viridis')

    plt.title('LDA Projection of Data')
    plt.legend()
    plt.show()


def LDAPlusSVM(input_path):
    data = []
    labels = []
    for cls, folder_name in enumerate(os.listdir(input_path)):
        folder_path = os.path.join(input_path, folder_name)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_vector = image.flatten()
            data.append(image_vector)
            labels.append(cls)

    X = np.array(data)
    y = np.array(labels)

    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建LDA模型并训练
    lda = LinearDiscriminantAnalysis()
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    # 创建SVM分类器并训练
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train_lda, y_train)
    # 可视化LDA后的特征
    # for i in range(6):
    #     visible_feature(X_train_lda, y_train, svm, i)

    # 使用SVM进行预测
    y_pred = svm.predict(X_test_lda)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"准确度: {accuracy}")
    print(f"混淆矩阵:\n{conf_matrix}")
    print(f"分类报告:\n{classification_rep}")


if __name__ == "__main__":
    input_folder = "sy1\\dataset"
    output_folder = "sy1\\redataset"
    output_folders = ["reclass1", "reclass2", "reclass3", "reclass4"]
    RGB2HSV(input_folder, output_folders)
    LDAPlusSVM(input_folder)
    LDAPlusSVM(output_folder)
