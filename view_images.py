from PIL import Image
import matplotlib.pyplot as plt
import os

normal_folder = "dataset/chest_xray/train/NORMAL"
pneumonia_folder = "dataset/chest_xray/train/PNEUMONIA"

normal_img = Image.open(
    os.path.join(normal_folder, os.listdir(normal_folder)[0])
)

pneumonia_img = Image.open(
    os.path.join(pneumonia_folder, os.listdir(pneumonia_folder)[0])
)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(normal_img, cmap="gray")
plt.title("Normal")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(pneumonia_img, cmap="gray")
plt.title("Pneumonia")
plt.axis("off")

plt.show()