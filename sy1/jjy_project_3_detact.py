from matplotlib import pyplot as plt
from jjy_project_2 import Model
import torch
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt



if __name__ == '__main__':
    # 加载模型参数
    model = Model()
    model.load_state_dict(torch.load('sy1/my_model_params.pth'))
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((60, 80)),
        transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图像
        transforms.ToTensor()
        ])
    image = cv2.imread("sy1/redataset/reclass3/32.jpg")
    image = transform(image)
    
    x = image.unsqueeze(0)

    for index, (name, layer) in enumerate(model.named_children()):
        f = 0
        x = layer(x)
        # 获取x的通道数
        if x.dim() == 2:
            x = x.view(x.size(0), x.size(1), 1, 1)
            f = 1
        num_channels = x.size(1)
        print(x.shape)

        # 计算行数和列数，每行最多显示5个子图
        num_cols = (num_channels + 15) // 16  # 向上取整除法，计算列数
        num_rows = min(16, num_channels)  # 每行最多显示5个子图

        # 创建一个图像，包含num_channels个子图
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(2, 12))

        x_cp = x.detach().cpu().numpy()
        x_cp = (x_cp - x_cp.min()) * 255 / (x_cp.max() - x_cp.min())
        print(x_cp)

        # 遍历每个通道，并显示单独的子图
        for channel in range(num_channels):
            row = channel % num_rows
            col = channel // num_rows
            channel_image = x_cp[0, channel]

            if num_cols == 1:
                axes[row].imshow(channel_image, cmap='gray', vmin=0, vmax=255)
                axes[row].set_xticks([])
                axes[row].set_yticks([])
            else:
                axes[row, col].imshow(channel_image, cmap='gray', vmin=0, vmax=255)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

        # 隐藏未使用的子图
        for channel in range(num_channels, num_rows * num_cols):
            row = channel % num_rows
            col = channel // num_rows
            if num_cols == 1:
                axes[row].axis('off')
            else:
                axes[row, col].axis('off')

        plt.show()

        if index == 2:
            x = x.view(x.size(0), -1)

        if f == 1:
            x = x.view(x.size(0), x.size(1))

    print(x)
