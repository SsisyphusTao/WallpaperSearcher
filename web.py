import remi.gui as gui
from remi import start, App

from pymongo import MongoClient
import torch

class WallpaperSearcher(App):
    def __init__(self, *args):
        super().__init__(*args)
        client = MongoClient('mongodb://127.0.0.1', 27017)
        db = client.wallhaven_v2
        self.wallpaper = db.wallpaper
        self.papers = list(self.wallpaper.find())
        self.outputs = db.outputs

    def main(self):
        background = gui.VBox()
        container = gui.VBox(width=200, height=100)

        self.input = gui.TextInput(hint='Image ID(0-10944):')
        self.bt = gui.Button('Search')

        # setting the listener for the onclick event of the button
        self.bt.onclick.do(self.on_button_pressed)

        # appending a widget to another, the first argument is a string key
        container.append(self.input)
        container.append(self.bt)

        compare = gui.HBox()
        query = gui.VBox()
        best = gui.VBox()
        self.query = gui.Image('https://hdwallpaperim.com/wp-content/uploads/2017/09/16/57315-wallhaven-Adobe_Photoshop-748x499.jpg',width=750, height=750)
        self.best  = gui.Image('https://hdwallpaperim.com/wp-content/uploads/2017/09/16/57315-wallhaven-Adobe_Photoshop-748x499.jpg',width=750, height=750)
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
        img_index = self.input.get_value()
        if img_index == '':
            return
        record = self.papers[int(img_index)]
        self.query.set_image(record['url'])
        self.query_url.set_text('Query URL: %s'%record['url'])

        record = self.outputs.find_one({'uid':record['uid']})
        query = torch.Tensor(record['img_embedding'])

        best =[0, '']
        for i in self.outputs.find():
            score = torch.nn.functional.cosine_similarity(query, torch.Tensor(i['img_embedding'])).mean()
            if score > best[0] and not score == 1:
                best[0] = score
                best[1] = i['uid']
        img_url = self.wallpaper.find_one({'uid':best[1]})['url']
        self.best.set_image(img_url)
        self.best_score.set_text('Best Match: %.6f  URL: %s'%(best[0], img_url))

if __name__ == '__main__':
    # starts the web server
    start(WallpaperSearcher, address='0.0.0.0' ,port=9090)