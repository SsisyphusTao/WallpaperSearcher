import bs4
from os import listdir
from pymongo import MongoClient

client = MongoClient('mongodb://127.0.0.1', 27017)
db = client.wallhaven_v2
papers = db.wallpaper

for i in listdir('/home/chandler/dataset/html/'):
    with open('/home/chandler/dataset/html/'+i, 'r') as f:
        soup = bs4.BeautifulSoup(f.read(), 'lxml')
    
    [img_info] = soup.find_all(id='wallpaper')
    tags = soup.find_all(**{'class':'tagname'})

    record = {}
    record['uid'] = i.split('.')[0]
    record['url'] = img_info.get('src')
    record['width'] = img_info.get('data-wallpaper-width')
    record['height'] = img_info.get('data-wallpaper-height')
    record['name'] = record['url'].split('/')[-1]
    record['Downloaded'] = False
    record['tags'] = {}

    for i in tags:
        record['tags'][i.text] = i.get('title')
    
    record['original_path'] = '/home/chandler/dataset/origin/%s'%record['name']
    record['train_path'] = '/home/chandler/dataset/train/%s.png'%record['uid']
    print(record)
    break
    # papers.insert_one(record)