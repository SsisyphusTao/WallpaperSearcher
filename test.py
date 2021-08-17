import torch
import cv2 as cv
import numpy as np
from resnet18 import get_model

from pymongo import MongoClient

model = get_model().cuda()
model.load_state_dict(torch.load('/home/chandler/towhee/checkpoints/ebd_019_0.289174.pth'))
model.eval()

client = MongoClient('mongodb://127.0.0.1', 27017)
db = client.Wallhaven
papers = db.wallpaper
vectors= db.tags_map
dataset= db.web_dataset

outputs = db.outputs

with torch.no_grad():
    for i in dataset.find():
        img = cv.imread(i['img_path'])
        h, w, _ = img.shape
        if h>w:
            d = (h-w)//2
            img = cv.copyMakeBorder(img,0,0,d,d,cv.BORDER_CONSTANT,value=(0,0,0))
        elif h<w:
            d = (w-h)//2
            img = cv.copyMakeBorder(img,d,d,0,0,cv.BORDER_CONSTANT,value=(0,0,0))
        img = cv.resize(img, (512,512))

        _, embedding = model(torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()/255.)
        record = {'img_path':i['img_path'], 'img_index':i['img_index'], 'img_embedding':embedding.cpu().numpy().tolist(), 'img_url':i['img_url']}
        outputs.insert_one(record)