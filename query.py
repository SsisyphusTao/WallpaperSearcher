import cv2 as cv
import torch
from sys import argv
from pymongo import MongoClient
from resnet18 import get_model

client = MongoClient('mongodb://127.0.0.1', 27017)
db = client.Wallhaven
outputs = db.outputs2

wallpapers = []
best = [0, '']

if '.jpg' in argv[1]:
    model = get_model().cuda()
    model.load_state_dict(torch.load('/home/chandler/towhee/checkpoints/ebd_019_0.289174.pth'))
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
    query = outputs.find_one({'img_index':int(argv[1])})
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