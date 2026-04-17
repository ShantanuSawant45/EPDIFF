import kagglehub
import os
import shutil

# Download dataset
path = kagglehub.dataset_download("dschettler8845/brats-2021-task1")
print("Downloaded at:", path)

# Create target directory inside EPDiff
target = "./data/BraTS21"
os.makedirs(target, exist_ok=True)

# Copy files into project
shutil.copytree(path, target, dirs_exist_ok=True)

print("Dataset copied to:", target)
