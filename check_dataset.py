import os

train_normal = "dataset/chest_xray/train/NORMAL"
train_pneumonia = "dataset/chest_xray/train/PNEUMONIA"

print("Normal Images:", len(os.listdir(train_normal)))
print("Pneumonia Images:", len(os.listdir(train_pneumonia)))