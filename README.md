# 📘 DBMS AI Teaching – Personal Study Assistant

This project is a **personal AI-powered study assistant** built to help me efficiently learn and revise **Database Management Systems (DBMS)** using my course lecture videos.

Instead of rewatching full lectures, this tool lets me:
- Ask DBMS-related questions
- Find **which lecture** covers the topic
- Get **exact timestamps (hours & minutes)** for quick revision

---

## 🎯 Purpose

This project is created **for my personal DBMS studies**.

It helps me:
- Revise topics quickly before exams
- Locate exact lecture segments
- Avoid wasting time searching through long videos
- Learn concepts with AI-assisted explanations

This is **not a generic chatbot**.  
It answers **only questions related to my DBMS course**.

---

## 🧠 How It Works 

1. Lecture text are converted into embeddings
2. User question is converted into an embedding
3. **Cosine similarity** finds the most relevant lecture chunks
4. A ** LLM ** explains:
   - Which lecture covers the topic
   - Exact timestamps to watch

All processing runs **locally**.

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** – UI
- **Ollama** – Local LLM & embeddings
- **nomic-embed-text** – Embeddings model
- **llama3.2** – Text generation
- **Scikit-learn** – Cosine similarity
- **Pandas ,NumPy and Requests**
- **Joblib** – Embedding storage

---

## ✨ Features

- 📚 DBMS-only question answering
- 🎥 Lecture-wise retrieval
- ⏱ Timestamp conversion (hours & minutes)
- 🧠 AI-generated human-like explanations
- 🖥 Fully offline & free 
- 📊 Streamlit-based clean UI

