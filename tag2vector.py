from pymongo import MongoClient
from torch._C import device
from transformers import AutoTokenizer, AutoModel, logging
import numpy as np
from tqdm import tqdm
import torch
import requests
import cv2 as cv

client = MongoClient('mongodb://127.0.0.1', 27017)
db = client.Wallhaven
papers = db.wallpaper
vectors= db.tags_map
dataset= db.dataset

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def transfer_tags():
    logging.set_verbosity_error()

    # Sentences we want sentence embeddings for
    sentences = ['dog cat', 'bird monkey']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')

    all_tags = []

    for i in papers.find():
        tags, _ = zip(*i['tags'])
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
            vectors.insert_one(record)
            bar.update(1)
    
def tags2vectors():
    tagMap = {}

    for n, i in enumerate(vectors.find()):
        tagMap[i['tags']]=[n, i['embedding']]
        # print(i['tags'])
    for i in papers.find():
        tags, _ = zip(*i['tags'])
        print(set(tags))
        tags = list(tags)
        tags.sort()
        ids, vecs = zip(*list(map(lambda x:tagMap[x],tags)))
        print(ids)
        print(np.array(vecs).shape)
        print(i['img_url'])
    print(len(tagMap))

def createDataset():
    tagMap = {}
    num = 0
    save_path = '/home/chandler/dataset/%06i.png'
    for n, i in enumerate(vectors.find()):
        tagMap[i['tags']]=[n, i['embedding']]
    for i in papers.find():
        tags, _ = zip(*i['tags'])
        tags = list(tags)
        tags.sort()
        ids, vecs = zip(*list(map(lambda x:tagMap[x],tags)))

        r = requests.get(i['img_url'])
        buffer = np.frombuffer(r.content,dtype=np.uint8)
        img = cv.imdecode(buffer,cv.IMREAD_COLOR)
        cv.imwrite(save_path%num, img)

        record = {'img_index':num, 'img_path':save_path%num, 'embedding_ids': ids, 'embeddings':vecs}
        dataset.insert_one(record)
        num += 1                    

if __name__ == '__main__':
    tags2vectors()