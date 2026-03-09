# 🎬 Movie ML Recommender System

A content-based Machine Learning movie recommendation system built entirely
in Python, trained on the TMDB 5000 Movies dataset from Kaggle. The system
applies TF-IDF vectorisation, KMeans clustering, Principal Component Analysis,
and cosine similarity to deliver intelligent movie recommendations, genre-based
browsing, Bayesian weighted ratings, and 5 data visualisation charts — all
through an interactive command-line interface.

---

## Features

- 🔍 Movie recommendations by title using cosine similarity
- 🎭 Genre-based browsing with Bayesian weighted ranking
- 🤖 KMeans clustering of 4,803 movies into 8 thematic groups
- ⭐ Statistically fair Bayesian weighted rating system
- 📊 5 Matplotlib charts — genre distribution, rating histogram,
     movies per year, top 10 rated, and PCA cluster scatter plot
- 💻 Interactive CLI menu — no technical knowledge required to use

---

## Tech Stack

| Library        | Purpose                                      |
|----------------|----------------------------------------------|
| pandas         | Data loading, cleaning, and preprocessing    |
| scikit-learn   | TF-IDF, KMeans, PCA, cosine similarity       |
| matplotlib     | All 5 data visualisation charts              |
| numpy          | Matrix and array operations                  |
| scipy          | Sparse matrix handling                       |

---

## Dataset

TMDB 5000 Movie Metadata — Kaggle
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Download and place tmdb_5000_movies.csv in the project folder.

---

## Installation

pip install pandas matplotlib scikit-learn numpy scipy

---

## Usage

python main.py

---

## Project Structure

MovieML/
├── main.py                     ← main program
├── tmdb_5000_movies.csv        ← dataset (download from Kaggle)
├── 01_genre_distribution.png   ← generated chart
├── 02_rating_distribution.png  ← generated chart
├── 03_movies_per_year.png      ← generated chart
├── 04_top_rated.png            ← generated chart
└── 05_clusters.png             ← generated chart

---

## ML Pipeline

CSV Input → Clean & Parse → TF-IDF Vectors →
PCA + KMeans → Cosine Similarity → Recommend & Visualise

---

## License

MIT License — free to use, modify, and distribute.
