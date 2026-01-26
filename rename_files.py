import os

folder = "./cont_val_circle"
for filename in os.listdir(folder):
    os.rename(os.path.join(folder, filename), os.path.join(folder, filename.replace("square", "circle")))
