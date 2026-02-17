import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests


st.set_page_config(
    page_title="DBMS AI Teaching",
    page_icon="📶",
    layout="wide"
)


st.sidebar.title("📶 DBMS AI Teaching")
st.sidebar.markdown("""

### About :
This is an **AI-powered DBMS course assistant**.

It helps you:
- Ask questions from the DBMS course
- Find **which lecture** covers the topic
- Get **exact timestamps** (hours & minutes)

---

### How it works
1. Your question is converted into embeddings  
2. Relevant lecture chunks are retrieved  
3. AI explains **where and how much** is taught  

""")
st.sidebar.markdown("---")
st.sidebar.caption("### Built for learning & revision ")



with st.sidebar.expander("Why I built this"):
    st.write(
        "I created this app for **my own DBMS learning**.\n\n"
        "- Faster revision before exams\n"
        "- Locate exact lecture timestamps\n"
        "- Avoid rewatching full videos\n"
        
    )


@st.cache_resource
def load_data():
    return joblib.load("all_embeddings.joblib")

df = load_data()

def creation_emb(text):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        }
    )
    data = r.json()
    return data["embeddings"]

def generate_llm(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    return r.json()["response"]

st.title("📊 DBMS Course Assistant")

query = st.text_input("Ask a question : ")

if st.button("Search"):
    with st.spinner("Searching relevant lectures..."):
        query_vector = creation_emb([query])[0]

        similarities = cosine_similarity(np.vstack(df["vectors"]),[query_vector]).flatten()

        best_res = similarities.argsort()[::-1][:5]
        new = df.loc[best_res, ["start", "end", "title", "text", "number"]]

        prompt =  f'''This is an  DBMS  course Zero to END. Here are video subtitle chunks containing video title, video number, start time in (hrs,min), end time in (min Hrs), the text at that time 
                    {new[["title", "number", "start", "end", "text"]].to_json(orient="records")} give timestamp in min and hrs only not in seconds everywehre 
                    ---------------------------------
                    {query} 
                    User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp in (min,hrs)) and guide the user to go to that particular video. 
                    If user asks unrelated question, tell him that you can only answer questions related to the course  Gvie answers like Human.
                    '''

        answer = generate_llm(prompt)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Relevant Video Segments")
    for _, row in new.iterrows():
        st.markdown(
            f"""
            **🥃 {row['title']} (Lecture {row['number']})**  
            ⏱ {row['start']//60}min → {(row['end']//60)+4}min
            """
                    )
