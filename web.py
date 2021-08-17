# import remi.gui as gui
# from remi import start, App



# class MyApp(App):
#     def __init__(self, *args):
#         super(MyApp, self).__init__(*args)

#     def main(self):
#         container = gui.VBox(width=120, height=100)
#         self.lbl = gui.Label('Hello world!')
#         self.bt = gui.Button('Press me!')

#         # setting the listener for the onclick event of the button
#         self.bt.onclick.do(self.on_button_pressed)

#         # appending a widget to another, the first argument is a string key
#         container.append(self.lbl)
#         container.append(self.bt)

#         # returning the root widget
#         return container

#     # listener function
#     def on_button_pressed(self, widget):
#         self.lbl.set_text('Button pressed!')
#         self.bt.set_text('Hi!')

# starts the web server
# start(MyApp)

import cv2 as cv
import torch
from pymongo import MongoClient

client = MongoClient('mongodb://127.0.0.1', 27017)
db = client.Wallhaven
outputs = db.outputs

wallpapers = []
best = [0, '']

query = outputs.find_one({'img_index':5})
tquery = torch.Tensor(query['img_embedding'])
img = cv.imread(query['img_path'])
cv.imwrite('query.jpg', cv.resize(img, (img.shape[1]//3, img.shape[0]//3)))
for i in outputs.find():
    # wallpapers.append([torch.nn.functional.cosine_similarity(tquery, torch.Tensor(i['img_embedding'])).mean(), i['img_path']])
    score = torch.nn.functional.cosine_similarity(tquery, torch.Tensor(i['img_embedding'])).mean()
    if score > best[0] and not score == 1:
        best[0] = score
        best[1] = i['img_path']
    # if wallpapers[-1][0] > 0.99:
    #     print(i['img_index'], wallpapers[-1][0], wallpapers[-1][1])
    #     cv.imwrite('%i.jpg'%i['img_index'], cv.imread(i['img_path']))
img = cv.imread(best[1])
cv.imwrite('best.jpg', cv.resize(img, (img.shape[1]//3, img.shape[0]//3)))