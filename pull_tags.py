import requests
from os import listdir
import json
from pymongo import MongoClient
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED, wait
from retrying import retry
import time

all_tags = set()
client = MongoClient('mongodb://172.17.0.1', 27017)
db = client.wallpaper
old_train_set = db.train_set
trainset = db.trainset
path = '/root/towhee/kaiyuan/dataset/train'

def pull_one(uid):
    content = requests.get('https://wallhaven.cc/api/v1/w/%s'%uid, timeout=10)
    data = json.loads(content.text)['data']
    for j in [x['name'] for x in data['tags']]:
        # all_tags.add(j)
    
        record = {}
        record['uid'] = uid
        record['width'] = data['dimension_x']
        record['height'] = data['dimension_y']
        record['tags'] =  [x['name'] for x in data['tags']]
        record['path'] = '%s/%s' % (path, i)
        # print(record)
        # break
        trainset.update_one({'uid':uid}, {'$setOnInsert':record}, upsert=True)


pool = []
losts = []

with open('losts.json', 'r') as f:
    uids = json.load(f)

with ThreadPoolExecutor(20) as executor:
    for i in tqdm(uids):
        # uid = i.split('.')[0]
        uid = i
        if not trainset.find_one({'uid':uid}):
            # trainset.update_one({'uid':uid}, {'$setOnInsert':old_train_set.find_one({'uid':uid})}, upsert=True)
            # continue
            losts.append(uid)
            pool.append(executor.submit(pull_one, uid))
            if pool.__len__() > 20:
                wait(pool, return_when=ALL_COMPLETED)
                pool.clear()
            time.sleep(5)
    wait(pool, return_when=ALL_COMPLETED)

with open('losts.json', 'w') as f:
    json.dump(list(losts), f)

