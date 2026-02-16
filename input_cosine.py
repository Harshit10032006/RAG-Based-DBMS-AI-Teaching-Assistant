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


simiralrities=cosine_similarity(np.vstack(df['vectors']),[query_vector]).flatten() # of all the vectors in the datafarmae (video)
# print(simiralrities)

n_results=5
best_res=simiralrities.argsort()[::-1][0:n_results] # sort the similartiy and give only top 3 indexs 

new = df.loc[best_res, ['start', 'end', 'title', 'text','number']]


prompt=f""" I will Give you an json file with start,end,Title,text,Number user asked an question  below give answer in format :
-The time stamp (which is in sec convert them in hrs, Minutes ) of the chunks with their title ,number+1(alaways) and the most related Text .
the chunks are :{new.to_json()}
STUDENT QUESTION: {query}
Dont give any extra text 
Answer like a friendly, patient teacher. Use bullet points, numbered steps, code examples, or tables if it helps explain better. Be clear and encouraging."""
 
# for index,item in new.iterrows():  # iterrows rows by rows 
#     print(item['text'],item['title'],item['start'],item['end'])

with open('prompt.txt','w',encoding="utf-8") as f :
    f.write(prompt)


