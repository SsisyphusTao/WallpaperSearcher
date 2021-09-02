import remi.gui as gui
from remi import start, App

from pymongo import MongoClient
import cv2 as cv
import requests
import numpy as np
import base64
import json
from pymilvus import connections, Collection

client = MongoClient('mongodb://127.0.0.1', 27017)
db = client.wallhaven_v3
wallpaper = db.wallpaper
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

class WallpaperSearcher(App):
    def __init__(self, *args):
        super().__init__(*args)
        self.on_process = False

    def main(self):
        background = gui.VBox()
        container = gui.VBox(width=200, height=100)

        self.input = gui.TextInput(hint='Image ID(0-10944)',width=1000)
        self.bt = gui.Button('Search')

        # setting the listener for the onclick event of the button
        self.bt.onclick.do(self.on_button_pressed)

        # appending a widget to another, the first argument is a string key
        container.append(self.input)
        container.append(self.bt)

        compare = gui.VBox()
        query = gui.VBox()
        best = gui.VBox()
        self.query = gui.Image('https://hdwallpaperim.com/wp-content/uploads/2017/09/16/57315-wallhaven-Adobe_Photoshop-748x499.jpg',width=512, height=512)
        self.best  = gui.Image(width=2048, height=512)
        self.best_score = gui.Label('Best')
        self.query_url = gui.Label('Query')

        query.append(self.query_url)
        query.append(self.query)
        best.append(self.best_score)
        best.append(self.best)

        compare.append(query)
        compare.append(best)

        background.append(container)
        background.append(compare)

        # returning the root widget
        return background

    # listener function
    def on_button_pressed(self, widget):
        if self.on_process:
            return
        self.on_process = True

        try:
            uid = n2uid[self.input.get_value()]
        except:
            self.on_process = False
            return
        print(uid)
        with open('/home/chandler/dataset/train/%s.png'%uid, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        self.query.set_image("data:image/jpeg;base64,%s"%img_data)
        record = outputs.find_one({'uid':uid})
        query = record['img_embedding']
      
        topK = 5
        search_params = {"metric_type": "IP"}
        # define output_fields of search result
        res = collection.search(
            query, "embedding", search_params, topK, output_fields=["nid"]
        )

        # show result
        best_imgs = []
        for hits in res:
            for hit in hits:
                best_match = n2uid[str(hit.entity.get("nid"))]
                best_imgs.append(cv.imread(wallpaper.find_one({'uid':best_match})['train_path']))
        _, jpg_data = cv.imencode('.jpg', cv.hconcat(best_imgs[1:]))
        img_data = base64.b64encode(jpg_data).decode("utf-8")
        self.best.set_image("data:image/jpeg;base64,%s"%img_data)
        self.on_process = False

if __name__ == '__main__':
    # starts the web server
    start(WallpaperSearcher, address='0.0.0.0' ,port=9090)