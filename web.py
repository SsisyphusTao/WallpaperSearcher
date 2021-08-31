import remi.gui as gui
from remi import start, App

from pymongo import MongoClient
import torch
from resnet18 import get_model
import cv2 as cv
import requests
import numpy as np
import base64

client = MongoClient('mongodb://127.0.0.1', 27017)
db = client.wallhaven_v3
wallpaper = db.wallpaper
papers = list(wallpaper.find())
outputs = db.outputs
model = get_model().cuda()
model.load_state_dict(torch.load('/home/chandler/towhee/checkpoints/v3_10epochs_010_0.0862574.pth'))
model.eval()

class WallpaperSearcher(App):
    def __init__(self, *args):
        super().__init__(*args)
        self.on_process = False

    def main(self):
        background = gui.VBox()
        container = gui.VBox(width=200, height=100)

        self.input = gui.TextInput(hint='Image ID(0-10944) or Image URL',width=1000)
        self.bt = gui.Button('Search')

        # setting the listener for the onclick event of the button
        self.bt.onclick.do(self.on_button_pressed)

        # appending a widget to another, the first argument is a string key
        container.append(self.input)
        container.append(self.bt)

        compare = gui.HBox()
        query = gui.VBox()
        best = gui.VBox()
        self.query = gui.Image('https://hdwallpaperim.com/wp-content/uploads/2017/09/16/57315-wallhaven-Adobe_Photoshop-748x499.jpg',width=512, height=512)
        self.best  = gui.Image('https://hdwallpaperim.com/wp-content/uploads/2017/09/16/57315-wallhaven-Adobe_Photoshop-748x499.jpg',width=512, height=512)
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
        img_index = self.input.get_value()
        if img_index == '':
            return
        uid = None
        try:
            url = papers[int(img_index)]['url']
            uid = papers[int(img_index)]['uid']
        except:
            url = self.input.get_value()

        self.query_url.set_text('Query URL: %s'%url)

        if uid:
            with open('/home/chandler/dataset/train/%s.png'%uid, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            self.query.set_image("data:image/jpeg;base64,%s"%img_data)
            record = outputs.find_one({'uid':uid})
            query = torch.Tensor(record['img_embedding'])
        else:
            self.query.set_image(url)
            r = requests.get(url)
            buffer = np.frombuffer(r.content,dtype=np.uint8)
            img = cv.imdecode(buffer,cv.IMREAD_COLOR)
            h, w, _ = img.shape
            if h>w:
                d = (h-w)//2
                img = cv.copyMakeBorder(img,0,0,d,d,cv.BORDER_CONSTANT,value=(0,0,0))
            elif h<w:
                d = (w-h)//2
                img = cv.copyMakeBorder(img,d,d,0,0,cv.BORDER_CONSTANT,value=(0,0,0))
            img = cv.resize(img, (512,512))

            _, tquery = model(torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()/255.)
            query = tquery.mean(-1).cpu()

        best =[0, '']
        s = torch.nn.functional.cosine_similarity(query, query).mean()
        for i in outputs.find():
            score = torch.nn.functional.cosine_similarity(query, torch.Tensor(i['img_embedding'])).mean()
            if score > best[0] and not score == s.item():
                best[0] = score
                best[1] = i['uid']
        img_url = wallpaper.find_one({'uid':best[1]})['url']
        with open('/home/chandler/dataset/train/%s.png'%best[1], 'rb') as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        self.best.set_image("data:image/jpeg;base64,%s"%img_data)
        self.best_score.set_text('Best Match: %.6f  URL: %s'%(best[0], img_url))
        self.on_process = False

if __name__ == '__main__':
    # starts the web server
    start(WallpaperSearcher, address='0.0.0.0' ,port=9090)