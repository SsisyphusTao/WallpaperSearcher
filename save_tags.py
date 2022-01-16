import json
from transformers import AutoModel, AutoTokenizer

from pymongo import MongoClient
from tqdm import tqdm
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

all_tags = set()
client = MongoClient('mongodb://172.17.0.1', 27017)
db = client.wallpaper
old_train_set = db.train_set
trainset = db.trainset
tag_embeddings = db.tag_embeddings

'''Save all tags from mongo without repetiton'''

# tags = set()
# for i in trainset.find():
#     tags.update(set(i['tags']))
# print(len(tags))
# tags = list(tags)
# tags.sort()
# with open('tags.json', 'w') as f:
#     json.dump(tags, f)

'''Transfer tags to vector'''

# with open('tags.json', 'r') as f:
#     all_tags = json.load(f)
# print(len(all_tags))
# model = AutoModel.from_pretrained('facebook/bart-large-cnn')
# tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')

# for i in tqdm(all_tags):
#     # Tokenize sentences
#     encoded_input = tokenizer(i, padding=True, truncation=True, return_tensors='pt')

#     # Compute token embeddings
#     with torch.no_grad():
#         model_output = model(**encoded_input)

#     # Perform pooling. In this case, max pooling.
#     sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).numpy().tolist()

#     record = {'tags':i, 'embedding':sentence_embeddings}
#     tag_embeddings.insert_one(record)

'''Create dataset collection'''

# tagDict = {}

# for i in tag_embeddings.find():
#     tagDict[i['tags']]=i['embedding']
# for i in tqdm(trainset.find()):
#     if i['tags'] == []:
#         print(i['uid'])
#     elif 'embeddings' in i:
#         continue
#     else:
#         # print(i['tags'])
#         tags = i['tags']
#         tags.sort()
#         vecs =list(map(lambda x:tagDict[x],tags))
#         # print(torch.Tensor(vecs).shape)
#         trainset.update_one({'uid':i['uid']}, {'$set':{'embeddings':vecs}})

'''Update labels'''

for i in tqdm(trainset.find()):
    x = torch.Tensor(i['embeddings'])

    a = x.max(0).values
    b = x.min(0).values
    c = x.mean(0)
    # d = x.std(0)
    # e = x.norm(dim=0)
    f = x.median(0).values
    # g = x.logsumexp(0)

    x = torch.cat([a,b,c,f])
    trainset.update_one({'uid':i['uid']}, {'$set':{'label':x.numpy().tolist()}})
    # print(len(x.numpy().tolist()))