import cv2 as cv
import torch
from sys import argv
from pymongo import MongoClient
from resnet18 import get_model
import wget
import time
import json
from pymilvus import connections, Collection

client = MongoClient('mongodb://172.17.0.1', 27017)
db = client.wallhaven_v3
papers = db.wallpaper
outputs = db.outputs

connections.connect()
collection=Collection('WallpaperSearcher')

default_index = {"index_type": "FLAT", "params": {"nlist": 768}, "metric_type": "IP"}
print(f"\nCreate index...")
collection.create_index(field_name="embedding", index_params=default_index)
print(f"\nload collection...")
collection.load()

with open('n2uid.json', 'r') as f:
    n2uid = json.load(f)

uid = n2uid[argv[1]]
vector =outputs.find_one({'uid':uid})['img_embedding']
query = papers.find_one({'uid':uid})
cv.imwrite('query.jpg', cv.imread(query['train_path']))

# load and search
topK = 5
search_params = {"metric_type": "IP"}
start_time = time.time()
print(f"\nSearch...")
# define output_fields of search result
res = collection.search(
    vector, "embedding", search_params, topK, output_fields=["nid"]
)
end_time = time.time()

# show result
print('Query uid: %s'%query['train_path'])
best_imgs = []
for hits in res:
    for hit in hits:
        # Get value of the random value field for search result
        best_match = n2uid[str(hit.entity.get("nid"))]
        print(hit, best_match)

        best_imgs.append(cv.imread(papers.find_one({'uid':best_match})['train_path']))
        
print("search latency = %.4fs" % (end_time - start_time))
cv.imwrite('best.jpg', cv.hconcat(best_imgs))