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


prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:
{new[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
{query}
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

 
# for index,item in new.iterrows():  # iterrows rows by rows 
#     print(item['text'],item['title'],item['start'],item['end'])   

# with open('prompt.txt','w',encoding="utf-8") as f :
#     f.write(prompt)


def model(prompt):
    r=requests.post('http://localhost:11434/api/generate',json={
            "model": 'llama3.2',
            "prompt": prompt,
            "stream":False
            })
    result=r.json()
    return result

print(model(prompt))