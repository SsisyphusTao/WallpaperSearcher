from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel, logging
from tqdm import tqdm
import torch
import cv2 as cv

client = MongoClient('mongodb://172.17.0.1', 27017)
db = client.wallhaven_v3
papers = db.wallpaper
tag_dict= db.tag_dict

def transfer():
    for i in papers.find():
        img = cv.imread(i['original_path'])
        h, w, _ = img.shape
        if h>w:
            d = (h-w)//2
            img = cv.copyMakeBorder(img,0,0,d,d,cv.BORDER_CONSTANT,value=(0,0,0))
        elif h<w:
            d = (w-h)//2
            img = cv.copyMakeBorder(img,d,d,0,0,cv.BORDER_CONSTANT,value=(0,0,0))
        img = cv.resize(img, (512,512))
        cv.imwrite(i['train_path'], img)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def transfer_tags():
    logging.set_verbosity_error()

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')

    all_tags = []

    for i in papers.find():
        tags = list(i['tags'].keys())
        all_tags += tags
    all_tags = list(set(all_tags))
    all_tags.sort()
    with tqdm(total=len(all_tags)) as bar:
        for i in all_tags:
            # Tokenize sentences
            encoded_input = tokenizer(i, padding=True, truncation=True, return_tensors='pt')

            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling. In this case, max pooling.
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).numpy().tolist()

            record = {'tags':i, 'embedding':sentence_embeddings}
            tag_dict.insert_one(record)
            bar.update(1)
    
def tags2vectors():
    tagDict = {}

    for n, i in enumerate(tag_dict.find()):
        tagDict[i['tags']]=[n, i['embedding']]
        # print(i['tags'])
    for i in papers.find():
        tags = list(i['tags'].values())
        tags.sort()
        print(tags)
        ids, vecs = zip(*list(map(lambda x:tagDict[x],tags)))
        print(ids)
        print(torch.Tensor(vecs).max())
        print(i['url'])
    print(len(tagDict))

def updateDataset():
    tagDict = {}

    for n, i in enumerate(tag_dict.find()):
        tagDict[i['tags']]=[n, i['embedding']]
    for i in papers.find():
        tags = list(i['tags'].keys())
        tags.sort()
        ids, vecs = zip(*list(map(lambda x:tagDict[x],tags)))
        # print(ids, np.array(vecs).shape)
        papers.update_one({'uid':i['uid']}, {'$set':{'embedding_index': ids, 'embeddings':vecs}})

def check():
    for i in papers.find():
        x = torch.Tensor(i['embeddings'])
        a = x.max(0).values
        b = x.min(0).values
        c = x.mean(0)
        # d = x.std(0)
        # e = x.norm(dim=0)
        f = x.median(0).values
        # g = x.logsumexp(0)

        x = torch.cat([a,b,c,f])
        papers.update_one({'uid':i['uid']}, {'$set':{'label':x.numpy().tolist()}})
        # print(len(x.numpy().tolist()))

if __name__ == '__main__':
    check()