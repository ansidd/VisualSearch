# hello_milvus.py demonstrates the basic operations of PyMilvus, a Python SDK of Milvus.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and hybrid search on entities
# 6. delete entities by PK
# 7. drop collection
import time
import numpy as np
import json
import sys
import cv2
import torch
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

suf_path = "./../project/"


with open(suf_path+'./config.json') as f:
    config = json.load(f)

output_path = config['paths']['output']

with open(suf_path+"video_database.pkl", "rb") as f:
    video_info = pickle.load(f)

video_database ={
    'video_name' : [],
    'embeddings' : [],
    'frame_num' : []
}


for i in video_info:
    num_frames = len(i['embeddings'])
    video_database['video_name'] += [i['name']]*num_frames
    video_database['embeddings'] += i['embeddings']
    video_database['frame_num'] += i['frame_num']

df = pd.DataFrame(video_database)

img_path = sys.argv[1]
img = cv2.imread(img_path)



le = LabelEncoder()
df['video_id'] = le.fit_transform(df['video_name'])

df.to_csv("vid_db.csv")

video_names = list(set(list(df['video_name'])))
id_to_name = dict(zip(le.transform(video_names), video_names))

mtcnn = torch.load(suf_path+config['paths']['models']['mtcnn'])
mtcnn.eval()
resnet = torch.load(suf_path+config['paths']['models']['resnet'])
resnet.eval()

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 512

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("video_db")
print(f"Does collection hello_milvus exist in Milvus: {has}")

if has:
    utility.drop_collection(collection_name='video_db')

#################################################################################
# 2. create collection
# We're going to create a collection with 3 fields.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "pk"    |    Int64   |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|  "random"  |    Double  |                  |      "a double field"        |
# +-+------------+------------+------------------+------------------------------+
# |3|"embeddings"| FloatVector|     dim=8        |  "float vector with dim 8"   |
# +-+------------+------------+------------------+------------------------------+
fields = [
    FieldSchema(name="serial_no", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="video_id", dtype=DataType.INT64),
    FieldSchema(name="frame_num", dtype=DataType.INT64),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "DB for Reverse Image Search")

print(fmt.format("Create collection `video_db`"))
video_db = Collection("video_db", schema, consistency_level="Strong")

################################################################################
# 3. insert data
# We are going to insert 3000 rows of data into `hello_milvus`
# Data to be inserted must be organized in fields.
#
# The insert() method returns:
# - either automatically generated primary keys by Milvus if auto_id=True in the schema;
# - or the existing primary key field from the entities if auto_id=False in the schema.

print(fmt.format("Start inserting entities"))
rng = np.random.default_rng(seed=19530)
entities = [
    # provide the pk field because `auto_id` is set to False
    list(range(len(df['video_id']))),
    df['video_id'].values,  # field random, only supports list
    df['frame_num'].values,
    df['embeddings'].values,    # field embeddings, supports numpy.ndarray and list
]

insert_result = video_db.insert(entities)

#print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entites

################################################################################
# 4. create index
# We are going to create an IVF_FLAT index for hello_milvus collection.
# create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

video_db.create_index("embeddings", index)

################################################################################
# 5. search, query, and hybrid search
# After data were inserted into Milvus and indexed, you can perform:
# - search based on vector similarity
# - query based on scalar filtering(boolean, int, etc.)
# - hybrid search based on vector similarity and scalar filtering.
#

# Before conducting a search or a query, you need to load the data in `hello_milvus` into memory.
print(fmt.format("Start loading"))
video_db.load()

# -----------------------------------------------------------------------------
# search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))



try:
    img_cropped = mtcnn(img)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    vectors_to_search = img_embedding.detach().numpy()
except Exception as e:
    print("Error while detecting face in the image: {0}".format(e))

    
search_params = {
    "metric_type": "l2",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = video_db.search(vectors_to_search, "embeddings", search_params, limit=5, output_fields=["video_id"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, video_name: {id_to_name[hit.entity.get('video_id')]}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------

# 7. drop collection
# Finally, drop the hello_milvus collection
print(fmt.format("Drop collection `video_db`"))
utility.drop_collection("video_db")
