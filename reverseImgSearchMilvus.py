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
import pandas as pd
from tqdm import tqdm

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

model = "resnet50"

def create_db():
    suf_path = "./"


    with open(suf_path+'./config.json') as f:
        config = json.load(f)

    output_path = config['paths']['output']
    embeddings_train = np.load(suf_path+output_path+'embeddings_train.npy', allow_pickle=True)
    labels_w_path_train = np.load(suf_path+output_path+'labels_train.npy', allow_pickle=True)
    embeddings_shape = embeddings_train.shape

    labels_train = labels_w_path_train[:,0]
    id_to_label = dict(enumerate(list(set(labels_train))))
    label_to_id = dict(zip(id_to_label.values(),id_to_label.keys()))
    n_classes = len(set(labels_train))
    label_ids_train = [label_to_id[i] for i in labels_train]
    print("No. of unique classes: ", n_classes)
    print("lenth of id_to_label: ", len(id_to_label))
    print("Embeddings shape",embeddings_shape)

    fmt = "\n=== {:30} ===\n"
    
    num_entities, dim = embeddings_shape

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

    has = utility.has_collection("img_db")
    print(f"Does collection hello_milvus exist in Milvus: {has}")

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
        FieldSchema(name="label", dtype=DataType.INT64),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    schema = CollectionSchema(fields, "DB for Reverse Image Search")

    print(fmt.format("Create collection `img_db`"))
    img_db = Collection("img_db", schema, consistency_level="Strong")

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
        list(range(len(embeddings_train))),
        label_ids_train,  # field random, only supports list
        embeddings_train,    # field embeddings, supports numpy.ndarray and list
    ]

    insert_result = img_db.insert(entities)

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

    img_db.create_index("embeddings", index)

    ################################################################################
    # 5. search, query, and hybrid search
    # After data were inserted into Milvus and indexed, you can perform:
    # - search based on vector similarity
    # - query based on scalar filtering(boolean, int, etc.)
    # - hybrid search based on vector similarity and scalar filtering.
    #

    # Before conducting a search or a query, you need to load the data in `hello_milvus` into memory.
    print(fmt.format("Start loading"))
    img_db.load()

    train_paths = list(labels_w_path_train[:,1])
    return img_db, fmt, id_to_label, train_paths

def calc_correct_frac(true_label, pred_labels):
    no_correct_occurences = pred_labels.count(true_label)
    correct_frac = no_correct_occurences/len(pred_labels)
    return correct_frac


def search_test(img_path, img_db, fmt, id_to_label):

    search_latency_fmt = "search latency = {:.4f}s"

    suf_path = "./../project/"

    with open(suf_path+'./config.json') as f:
        config = json.load(f)

    output_path = config['paths']['output']
    embeddings_test = np.load(suf_path+output_path+'embeddings_test.npy', allow_pickle=True)
    labels_w_path_test = np.load(suf_path+output_path+'labels_test.npy', allow_pickle=True)
    embeddings_shape = embeddings_test.shape

    labels_test = labels_w_path_test[:,0]
    #id_to_label = dict(enumerate(list(set(labels_test))))
    #label_to_id = dict(zip(id_to_label.values(),id_to_label.keys()))
    #n_classes = len(set(labels_test))
    #label_ids_test = [label_to_id[i] for i in labels_test]

    search_params = {
        "metric_type": "l2",
        "params": {"nprobe": 10},
    }

    

    correct_fracs = []
    for index in tqdm(range(len(embeddings_test))):
        vectors_to_search = embeddings_test[index,:]
        vectors_to_search = vectors_to_search.flatten()
        vectors_to_search = np.expand_dims(vectors_to_search, 0)
        true_label = labels_test[index]

        start_time = time.time()
        result = img_db.search(vectors_to_search, "embeddings", search_params, limit=20, output_fields=["label"])
        end_time = time.time()
        res_labels = []
        for hits in result:
             for hit in hits:
                 res_labels.append(id_to_label[hit.entity.get('label')])
        #         print(f"hit: {hit}, label: {id_to_label[hit.entity.get('label')]}")

        #print(search_latency_fmt.format(end_time - start_time))
        #res_labels = pd.Series(res_labels).value_counts()
        #res_labels.index = pd.Series(res_labels.index).apply(lambda a: id_to_label[a])
        correct_fracs.append(calc_correct_frac(true_label, res_labels))
        #break

    print(np.mean(correct_fracs))



def search(img_path, img_db, fmt, id_to_label, train_paths):
    search_latency_fmt = "search latency = {:.4f}s"

    suf_path = "./../project/"

    with open(suf_path+'./config.json') as f:
        config = json.load(f)


    img = cv2.imread(img_path)

    if (model=='facenet'): 

        mtcnn = torch.load(suf_path+config['paths']['models']['mtcnn'])
        mtcnn.eval()
        resnet = torch.load(suf_path+config['paths']['models']['resnet'])
        resnet.eval()

        # -----------------------------------------------------------------------------
        # search based on vector similarity
        print(fmt.format("Start searching based on vector similarity"))
        
        try:
            img_cropped = mtcnn(img)
            img_embedding = resnet(img_cropped.unsqueeze(0))
            vectors_to_search = img_embedding.detach().numpy()
        except Exception as e:
            print("Error while detecting face in the image: {0}".format(e))
        
    else:
        output_path = config['paths']['output']
        embeddings_train = np.load(suf_path+output_path+'embeddings_resnet50_10testimg.npy', allow_pickle=True)
        labels_w_path_train = np.load(suf_path+output_path+'labels_resnet50_10testimg.npy', allow_pickle=True)
        indices = np.where(labels_w_path_train[:,1]==img_path)
        vectors_to_search = embeddings_train[indices[0]]
        print("Vecror shape",vectors_to_search.shape)



    search_params = {
        "metric_type": "l2",
        "params": {"nprobe": 10},
    }

    print(vectors_to_search.shape)
    start_time = time.time()
    result = img_db.search(vectors_to_search, "embeddings", search_params, limit=200, output_fields=["label", "serial_no"])
    end_time = time.time()

    res_labels = []
    res_arr = []
    last_embedding = None
    counter=0
    img_names = []
    for hits in result:
        for hit in hits:
            res_labels.append(hit.entity.get('label'))
            l, p = id_to_label[hit.entity.get('label')], train_paths[hit.entity.get('serial_no')]
            img_name = p.split("/")[-1]
            img_name = img_name.split("_")
            img_name = "_".join(img_name[:-1] + [img_name[-1][:4]])
            if img_name not in img_names:
                res_arr.append([l,p])
                counter+=1
                img_names.append(img_name)
            
            print(f"hit: {hit}, label: {id_to_label[hit.entity.get('label')]}")
            if counter==20:
                break

    print(img_names)
    print(search_latency_fmt.format(end_time - start_time))
    res_labels = pd.Series(res_labels).value_counts()
    res_labels.index = pd.Series(res_labels.index).apply(lambda a: id_to_label[a])
    print(res_labels)
    return res_arr
# -----------------------------------------------------------------------------

def drop_db(fmt):
    # 7. drop collection
    # Finally, drop the hello_milvus collection
    print(fmt.format("Drop collection `img_db`"))
    utility.drop_collection("img_db")

def main():
    img_path = sys.argv[1]
    img_db, fmt, id_to_label, train_paths= create_db()
    if img_path=='test':
        search_test(img_path, img_db, fmt, id_to_label)
    else:
        search(img_path, img_db, fmt, id_to_label, train_paths)
    drop_db(fmt)
    
if __name__=="__main__":
    main()