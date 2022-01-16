from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections
from pymongo import MongoClient

def createBaseCollection():
    connections.connect()
    print(f"\nList collections...")
    print(list_collections())


    default_fields = [
            FieldSchema(name="nid", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]

    default_schema = CollectionSchema(default_fields, '10944')


    print(f"\nCreate collection...")
    collection = Collection(name="WallpaperSearcher", schema=default_schema)

    print(f"\nList collections...")
    print(list_collections())

def insertData():
    import numpy as np
    import json
    client = MongoClient('mongodb://172.17.0.1', 27017)
    db = client.wallhaven_v3
    outputs = db.outputs
    dim = 768
    nb = 10944
    
    nid = []
    vectors = []
    n2uid = {}

    for n, i in enumerate(outputs.find()):
        n2uid[n] = i['uid']
        nid.append(n)
        temp=np.array(i['img_embedding'])
        temp /= temp.std()
        vectors += temp.tolist()
    
    print(np.array(nid).shape)
    print(np.array(vectors).shape)

    connections.connect()
    print(f"\nList collections...")
    print(list_collections())
    collection=Collection(*list_collections())
    collection.insert([nid, vectors])
    with open('n2uid.json', 'w') as f:
        json.dump(n2uid, f)

def dropData():
    connections.connect()
    print(f"\nList collections...")
    print(list_collections())
    collection=Collection(*list_collections())
    collection.drop()

def query():
    client = MongoClient('mongodb://172.17.0.1', 27017)
    db = client.wallhaven_v3
    outputs = db.outputs
    connections.connect()
    print(f"\nList collections...")
    print(list_collections())
    collection=Collection(*list_collections())

    default_index = {"index_type": "FLAT", "params": {"nlist": 768}, "metric_type": "IP"}
    print(f"\nCreate index...")
    collection.create_index(field_name="embedding", index_params=default_index)
    print(f"\nload collection...")
    collection.load()

    # load and search
    topK = 5
    search_params = {"metric_type": "IP"}
    import time
    start_time = time.time()
    print(f"\nSearch...")
    # define output_fields of search result
    q = outputs.find_one()
    res = collection.search(
        q['img_embedding'], "embedding", search_params, topK, output_fields=["nid"]
    )
    end_time = time.time()

    import json
    with open('n2uid.json', 'r') as f:
        n2uid = json.load(f)
    # show result
    print('Query uid: %s'%q['uid'])
    for hits in res:
        for hit in hits:
            # Get value of the random value field for search result
            print(hit, n2uid[str(hit.entity.get("nid"))])
    print("search latency = %.4fs" % (end_time - start_time))

if __name__ == '__main__':
    query()