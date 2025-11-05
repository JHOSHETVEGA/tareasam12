import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SISTEMA DE RECOMENDACIÓN BASADO EN CONTENIDO + VISUALIZACIÓN

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📘",
    layout="wide",
)


st.markdown("""
<style>
body {
    background-color: #f4f6fa;
}
.book-card {
    background-color: white;
    padding: 20px;
    margin-bottom: 15px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.title {
    color: #1E90FF;
    font-size: 22px;
    font-weight: bold;
}
.author {
    color: #34495E;
    font-style: italic;
    margin-bottom: 5px;
}
.genre {
    background-color: #D6EAF8;
    color: #154360;
    border-radius: 8px;
    padding: 3px 8px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---- Encabezado ----
st.title("Recomendación de libros ")
st.markdown("#### Basado en *TF-IDF + Cosine Similarity*.")
st.info("Upload a CSV file containing: `title`, `author`, `genre`, `description`.")

# CARGA DE ARCHIVO CSV
archivo = st.file_uploader("Upload your CSV file", type=["csv"])

if archivo is not None:
    try:
        df = pd.read_csv(archivo)

        columnas_requeridas = {"title", "author", "genre", "description"}
        if not columnas_requeridas.issubset(set(df.columns)):
            st.error(f"The file must contain the columns: {', '.join(columnas_requeridas)}")
        else:
            df["contenido"] = df["author"] + " " + df["genre"] + " " + df["description"]

            # Vectorización TF-IDF en inglés 
            tfidf = TfidfVectorizer(stop_words="english")
            matriz_tfidf = tfidf.fit_transform(df["contenido"])

            # Matriz de similitud coseno
            similaridad = cosine_similarity(matriz_tfidf)

            # Función recomendadora con puntajes
            def recomendar_libros(titulo, n=5):
                if titulo not in df["title"].values:
                    return pd.DataFrame()
                idx = df[df["title"] == titulo].index[0]
                scores = list(enumerate(similaridad[idx]))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                indices = [i[0] for i in scores[1:n+1]]
                valores = [i[1] for i in scores[1:n+1]]
                recomendaciones = df.iloc[indices][["title", "author", "genre", "description"]].copy()
                recomendaciones["score"] = valores
                return recomendaciones

            # Interfaz de selección
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                libro_seleccionado = st.selectbox("📘 Select a book:", df["title"].values)
            with col2:
                n_recomendaciones = st.slider("Number of recommendations", 1, 10, 5)

            if st.button(" Recommend"):
                recomendaciones = recomendar_libros(libro_seleccionado, n=n_recomendaciones)
                if recomendaciones.empty:
                    st.warning("No similar books found.")
                else:
                    st.markdown(f"###  Books similar to **{libro_seleccionado}**")
                    st.markdown("---")

                    # --- Gráfico de barras con puntajes ---
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(recomendaciones["title"], recomendaciones["score"], color="#1E90FF")
                    ax.set_xlabel("Similarity Score (Cosine)")
                    ax.set_ylabel("Book Title")
                    ax.invert_yaxis()
                    ax.set_title("Similarity Scores for Recommended Books")
                    st.pyplot(fig)

                    # --- Mostrar tarjetas de recomendaciones ---
                    for _, row in recomendaciones.iterrows():
                        st.markdown(f"""
                        <div class="book-card">
                            <div class="title">{row['title']}</div>
                            <div class="author">by {row['author']}</div>
                            <div class="genre">{row['genre']}</div>
                            <p><b>Similarity Score:</b> {row['score']:.3f}</p>
                            <p>{row['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f" Error reading file: {e}")
else:
    st.warning(" Waiting for you to upload your CSV file...")
