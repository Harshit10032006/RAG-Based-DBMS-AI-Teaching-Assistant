import requests 
import os 
import json
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib

def creation_emb(text):
    r=requests.post('http://localhost:11434/api/embed',json={
            'model':'nomic-embed-text',
            'input':text
    })

    data=r.json()
    
    if 'embeddings' not in data:
        raise RuntimeError(f"Ollama error: {data}")

    return data['embeddings']



cchunks=os.listdir('Chunks')

dictt=[]
id=0
for chunkse in cchunks :
    with open(f"Chunks/{chunkse}") as f:
        content=json.load(f)
    print('Creating Embedding !!!!')
    all_vectors=creation_emb([emb['text'] for emb in content['chunks']])
    for i,chunk in enumerate(content['chunks']):
        chunk['unique_id']=id
        chunk['vectors']=all_vectors[i]
        id+=1
        dictt.append(chunk)
        
     
            
    
    

df=pd.DataFrame().from_records(dictt)

joblib.dump(df,'all_embeddings.joblib')





