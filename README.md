# Hybrid Insurance Recommendation Engine 🧠

This project implements a **hybrid recommendation system** for insurance products using:

- User-based collaborative filtering (cosine similarity)
- A simple Neural Collaborative Filtering (NCF) model
- REST API to serve recommendations

---

## 🧱 Components

- `data/user_policy_ratings.csv` – synthetic user–policy rating data  
- `src/recommender.py` – collaborative filtering logic  
- `model/train_ncf.py` – trains a small neural CF model  
- `app/main.py` – FastAPI app exposing `/recommend`  

---

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- scikit-learn
- TensorFlow / Keras
- FastAPI

---

## 🚀 Setup

```bash
pip install -r requirements.txt
