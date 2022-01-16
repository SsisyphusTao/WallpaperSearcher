import time
import requests
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np
import cv2 as cv

client = MongoClient('mongodb://172.17.0.1', 27017)
db = client.wallhaven_v2
papers = db.wallpaper

bar = tqdm(total=papers.find().count())
one = papers.find_one({'Downloaded':False})

while not one == None:
    try:
        r = requests.get(one['url'], timeout=5)
    except:
        continue
    if r.status_code == 429:
        continue

    buffer = np.frombuffer(r.content,dtype=np.uint8)
    img = cv.imdecode(buffer,cv.IMREAD_COLOR)
    cv.imwrite(one['original_path'], img)
    papers.update_one({'uid':one['uid']}, {'$set':{'Downloaded':True}})
    bar.update(1)

    one = papers.find_one({'Downloaded':False})