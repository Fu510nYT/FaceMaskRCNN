import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 4
net = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = net.roi_heads.box_predictor.cls_score.in_features
net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
net.load_state_dict(torch.load("D:/workspace/NathanTrain/model.pth"))
net.eval()
net = net.to(device)

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32) / 255
    img = torch.from_numpy(img)
    img = torch.permute(img, (2, 0, 1))
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    # print(img.shape)
    h = net(img)
    # print(h)
    o_boxes = h[0]["boxes"]
    o_labels = h[0]["labels"]
    o_scores = h[0]["scores"]
    for box, label, score in zip(o_boxes, o_labels, o_scores):
        if score < 0.7: continue
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        if label == 2: color = (0, 0, 255)
        elif label == 3: color = (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_names = ["with mask", "without mask", "incorrect"]
        cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("frame", frame)
    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()