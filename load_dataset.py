import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from bs4 import BeautifulSoup


class MyDataset(Dataset):
    def __init__(self, root):
        self.root = root
        #print(root)
        self.images = list(sorted(os.listdir("D:/workspace/NathanTrain")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_image = "maksssksksss" + str(idx) + ".png"
        file_label = "maksssksksss" + str(idx) + ".xml"
        image_path = os.path.join(self.root, "images", file_image)
        #print(self.root)
        label_path = os.path.join(self.root, "annotations", file_label)

        x = cv2.imread(image_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.array(x, dtype=np.float32) / 255
        x = torch.from_numpy(x)
        x = torch.permute(x, (2, 0, 1))
        y = {}
        with open(label_path) as f:
            data = f.read()
            soup = BeautifulSoup(data, "xml")


            objs = soup.find_all("object")
            y["boxes"] = []
            y["labels"] = []
            y["image_id"] = torch.tensor([idx])
            for obj in objs:
                x1 = int(obj.find("xmin").text)
                y1 = int(obj.find("ymin").text)
                x2 = int(obj.find("xmax").text)
                y2 = int(obj.find("ymax").text)
                y["boxes"].append([x1, y1, x2, y2])
                label = 0
                if obj.find("name").text == "with_mask": label = 1
                if obj.find("name").text == "without_mask": label = 2
                if obj.find("name").text == "mask_weared_incorrect": label = 3
                y["labels"].append(label)
            y["boxes"] = torch.as_tensor(y["boxes"], dtype=torch.float32)   # data type
            y["labels"] = torch.as_tensor(y["labels"], dtype=torch.int64)
        return x, y