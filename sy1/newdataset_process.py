import os
import shutil

# 定义原始数据集文件夹路径和目标文件夹路径
source_dir = "sy1/MDSD_subset"
target_dir = "sy1/MDSM_subset_plus"

# 创建目标文件夹
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历原始数据集中的子文件夹
for subfolder in os.listdir(source_dir):
    subfolder_path = os.path.join(source_dir, subfolder)
    
    # 检查是否是文件夹
    if os.path.isdir(subfolder_path):
        # 创建目标子文件夹
        target_subfolder_path = os.path.join(target_dir, subfolder)
        if not os.path.exists(target_subfolder_path):
            os.makedirs(target_subfolder_path)
        
        # 遍历原始子文件夹中的类别文件夹
        for class_folder in os.listdir(subfolder_path):
            class_folder_path = os.path.join(subfolder_path, class_folder)
            
            # 检查是否是文件夹
            if os.path.isdir(class_folder_path):
                # 拷贝类别文件夹中的文件到目标子文件夹中
                target_class_folder_path = os.path.join(target_subfolder_path, class_folder)
                if not os.path.exists(target_class_folder_path):
                    os.makedirs(target_class_folder_path)
                
                # 遍历类别文件夹中的文件
                for file_name in os.listdir(class_folder_path):
                    file_path = os.path.join(class_folder_path, file_name)
                    target_file_path = os.path.join(target_class_folder_path, file_name)
                    
                    # 拷贝文件
                    shutil.copy(file_path, target_file_path)

print("拼接完成，文件已放置到MDSM_subset_plus文件夹中。")
