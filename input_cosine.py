import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests

df=joblib.load('all_embeddings.joblib')



def creation_emb(text):
    r=requests.post('http://localhost:11434/api/embed',json={
            'model':'nomic-embed-text',
            'input':text
    })

    data=r.json()
    
    if 'embeddings' not in data:
        raise RuntimeError(f"Ollama error: {data}")

    return data['embeddings']




query=input("ASK : ")
query_vector=creation_emb([query])[0]
# print(query_vector)


simiralrities=cosine_similarity(np.vstack(df['vectors']),[query_vector]).flatten()
print(simiralrities)

best_res=simiralrities.argsort()[::-1][0:3]

new=df.loc[best_res]
print(new[['title','text','number']])
