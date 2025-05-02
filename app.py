
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache_data
def load_data():
    return pd.read_csv('music_data.csv')

def build_recommender(df):
    df['combined_features'] = df['genre'] + " " + df['artist']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, cosine_sim, df):
    indices = pd.Series(df.index, index=df['title'])
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    music_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[music_indices]

st.title("🎧 RT7 Music Recommender System")
df = load_data()
cosine_sim = build_recommender(df)

selected_song = st.selectbox("Choose a song you like:", df['title'].values)

if st.button("Recommend Similar Songs"):
    recommendations = get_recommendations(selected_song, cosine_sim, df)
    if recommendations.empty:
        st.write("No similar songs found.")
    else:
        st.write("Here are some similar songs you might enjoy:")
        for rec in recommendations:
            st.markdown(f"- {rec}")
