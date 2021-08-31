import torch
import cv2 as cv
import numpy as np
from resnet18 import get_model

from pymongo import MongoClient

model = get_model().cuda()
model.load_state_dict(torch.load('/home/chandler/towhee/checkpoints/v2_100epochs_046_1.23216.pth'))
model.eval()

client = MongoClient('mongodb://127.0.0.1', 27017)
db = client.wallhaven_v2
papers = db.wallpaper
outputs = db.outputs

with torch.no_grad():
    for i in papers.find():
        img = cv.imread(i['train_path']).astype(np.float32)
        _, embedding = model(torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()/255.)
        record = {'uid':i['uid'], 'img_embedding':embedding.cpu().numpy().tolist()}
        outputs.insert_one(record)