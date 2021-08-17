#@Time :2021/4/12 21:20
#@file : SpiderWallhaven.py
'''代码爬取的是"https://wallhaven.cc/toplist?page=1"网页
关于15行正则表达式解析的是每张图片的源地址，(不去解析图片的源地址将得到每张图片的缩略图
只有解析它们的源地址，开展以下的爬取工作就交给循环)
'''
import requests
# import cv2 as cv
import bs4
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient

client = MongoClient('mongodb://127.0.0.1', 27017)
db = client.Wallhaven
collection = db.wallpaper

with tqdm(total=1000) as bar:
    page = 0
    while not bar.n == bar.total:#爬取目标的页数，步长为1，本次实验为2页
        page += 1
        url = "https://wallhaven.cc/toplist?page=%d"%page#page=？根据规律发现是从1迭代的，这里用占位符
        result = requests.get(url)#生成请求
        page_soup = bs4.BeautifulSoup(result.content.decode(),'lxml')
        for n, i in enumerate([x.get('href') for x in page_soup.select('a') if x.get('class') == ['preview']]):
            furl = requests.get(i)
            soup = bs4.BeautifulSoup(furl.content.decode(),'lxml')
            
            record = {}
            record['tags'] = [[x.get('title'), x.text] for x in soup.select('a') if x.get('class') == ['tagname']]

            for j in soup.select('img'):
                if j.get('id') == 'wallpaper':
                    record['img_url'] = j.get('src')
                    # resld = requests.get(j.get('src'))
                # with open('Wallhaven/%d.jpg' % count, 'wb') as f:
                #     f.write(resld.content)
                    # buffer = np.frombuffer(resld.content,dtype=np.uint8)
                    # img = cv.imdecode(buffer,cv.IMREAD_COLOR)
                    # img = cv.resize(img ,(img.shape[1]//4, img.shape[0]//4))
                    # cv.imshow('s', img)
                    # print("Have downloaded %d 张..."%count)
                    # if cv.waitKey() == ord('q'):
                        # exit()
            record['position'] = [page, n]
            collection.insert_one(record)
            bar.set_postfix(pos=record['position'])
            bar.update(1)