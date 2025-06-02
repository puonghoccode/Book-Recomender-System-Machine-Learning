import streamlit as st
import pickle
import numpy as np

# Title
st.title('📚 Book Recommender System')

# Load artifacts
model = pickle.load(open('artifacts/model.pkl','rb'))
books_tfidf = pickle.load(open('artifacts/books_tfidf.pkl','rb'))
book_names = pickle.load(open('artifacts/book_names.pkl','rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl','rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl','rb'))
cosine_sim = pickle.load(open('artifacts/cosine_sim.pkl','rb'))
books = pickle.load(open('artifacts/books.pkl','rb'))
book_clusters = pickle.load(open('artifacts/book_clusters.pkl','rb'))

# Fetch poster function
def fetch_poster(suggestion):
    poster_url = []
    for book_id in suggestion[0]:  # suggestion[0] là list index sách
        book_name = book_pivot.index[book_id]
        idx = np.where(final_rating['title'] == book_name)[0][0]
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)
    return poster_url

# Select method
method = st.selectbox("Select Recommendation Method", ["Item-based CF", "Content-based TF-IDF", "Clustering"])
if method == "Content-based TF-IDF":
    select_source = books['title'].unique()
else:
    select_source = book_names

# Select book title
book_name = st.selectbox("Select Book", book_names)

# Item-based CF
def recommend_item_based(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)
    poster_url = fetch_poster(suggestion)

    for book_id in suggestion[0]:
        books_list.append(book_pivot.index[book_id])

    return books_list, poster_url

# Content-based TF-IDF
def recommend_content_based(book_name):
    books_list = []
    poster_url = []

    idx = books_tfidf[books_tfidf['title'] == book_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:7]  # 6 books

    book_indices = [i[0] for i in sim_scores]

    for i in book_indices:
        books_list.append(books_tfidf['title'].iloc[i])
        url = books_tfidf['image_url'].iloc[i]
        poster_url.append(url)

    return books_list, poster_url

# Clustering
def recommend_clustering(book_name):
    books_list = []
    poster_url = []

    cluster_id = book_clusters[book_clusters['title'] == book_name]['cluster'].values[0]
    cluster_books = book_clusters[book_clusters['cluster'] == cluster_id]['title']

    # Lấy top 6 sách khác cuốn đang chọn
    count = 0
    for title in cluster_books:
        if title != book_name:
            books_list.append(title)
            idx = np.where(final_rating['title'] == title)[0][0]
            url = final_rating.iloc[idx]['image_url']
            poster_url.append(url)
            count += 1
            if count == 6:
                break

    return books_list, poster_url

# Show recommendation button
if st.button('Show Recommendation'):

    if method == "Item-based CF":
        recommended_books, poster_url = recommend_item_based(book_name)

    elif method == "Content-based TF-IDF":
        if book_name not in books_tfidf['title'].values:
            st.error("This book is not available in Content-based model. Please select another book.")
        else:
            recommended_books, poster_url = recommend_content_based(book_name)

    elif method == "Clustering":
        recommended_books, poster_url = recommend_clustering(book_name)

    # Display in 5 columns
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(recommended_books[i])
            st.image(poster_url[i])
