import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from load_dataset import MyDataset
import matplotlib.pyplot as plt


def collate_fn(batch): return tuple(zip(*batch))

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 4
net = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = net.roi_heads.box_predictor.cls_score.in_features
net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
if os.path.exists("model.pth"):
    net.load_state_dict(torch.load("model.pth"))
net = net.to(device)

batch_size = 1
train_data = MyDataset("./dataset_pl")
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True
)

writer = SummaryWriter("runs")
params = [p for p in net.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0005
)
start_from = 0
num_epochs = 100000000000000000
for e in range(start_from, start_from + num_epochs, 1):
    net.train()
    running_loss = 0
    for batch, (x, y) in enumerate(train_loader, 1):
        x = [i.to(device) for i in x]
        y = [{k: v.to(device) for k, v in i.items()} for i in y]

        loss_dict = net(x, y)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        print("Epoch %d(%d/%d): train_loss: %.4f" % (e, batch, len(train_loader), running_loss / batch))


        # plt.show()

    running_loss = running_loss / len(train_loader)
    writer.add_scalars("loss", {"train": running_loss}, e)
    writer.flush()
    print("Epoch %d: train_loss: %.4f" % (e, running_loss))

    torch.save(net.state_dict(), "model.pth")
