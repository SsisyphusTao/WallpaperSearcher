import cv2 as cv
import torch
from sys import argv
from pymongo import MongoClient
from resnet18 import get_model

client = MongoClient('mongodb://127.0.0.1', 27017)
db = client.wallhaven_v2
papers = db.wallpaper
outputs = db.outputs

wallpapers = []
best = [0, '', '']

if '.jpg' in argv[1]:
    model = get_model().cuda()
    model.load_state_dict(torch.load('/home/chandler/towhee/checkpoints/v2_100epochs_046_1.23216.pth'))
    model.eval()
    img = cv.imread(argv[1])
    h, w, _ = img.shape
    if h>w:
        d = (h-w)//2
        img = cv.copyMakeBorder(img,0,0,d,d,cv.BORDER_CONSTANT,value=(0,0,0))
    elif h<w:
        d = (w-h)//2
        img = cv.copyMakeBorder(img,d,d,0,0,cv.BORDER_CONSTANT,value=(0,0,0))
    img = cv.resize(img, (512,512))

    _, tquery = model(torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()/255.)
    tquery = tquery.cpu()
else:
    query = outputs.find()[int(argv[1])]
    tquery = torch.Tensor(query['img_embedding'])
    img_path = papers.find_one({'uid':query['uid']})['train_path']
    print(query['uid'])
    cv.imwrite('query.jpg', cv.imread(img_path))

s = 1-torch.nn.functional.pairwise_distance(tquery, tquery).mean()
for i in outputs.find():
    score = 1-torch.nn.functional.pairwise_distance(tquery, torch.Tensor(i['img_embedding'])).mean()
    score = score.item()
    if score > best[0] and not score == s.item():
        best[0] = score
        best[1] = papers.find_one({'uid':i['uid']})['train_path']
        best[2] = i['uid']
cv.imwrite('best.jpg', cv.imread(best[1]))
print(best[0], best[2])