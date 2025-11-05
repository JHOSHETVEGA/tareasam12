import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# 🎯 SISTEMA DE RECOMENDACIÓN BASADO EN CONTENIDO
# ==========================================================

st.set_page_config(
    page_title="Recomendador de Libros",
    page_icon="📚",
    layout="wide",
)

# ---- Estilos personalizados ----
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
st.title("📖 Sistema de Recomendación de Libros")
st.markdown("#### Basado en *TF-IDF + Similitud del Coseno* (filtrado por contenido).")
st.info("📂 Sube un archivo CSV que contenga las columnas: `title`, `author`, `genre`, `description`.")

# ==========================================================
# 📤 CARGA DE ARCHIVO CSV
# ==========================================================

archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    try:
        df = pd.read_csv(archivo)

        # Validar columnas necesarias
        columnas_requeridas = {"title", "author", "genre", "description"}
        if not columnas_requeridas.issubset(set(df.columns)):
            st.error(f"El archivo debe contener las columnas: {', '.join(columnas_requeridas)}")
        else:
            # Crear campo combinado de contenido
            df["contenido"] = df["author"] + " " + df["genre"] + " " + df["description"]

            # Vectorización TF-IDF
            tfidf = TfidfVectorizer(stop_words="spanish")
            matriz_tfidf = tfidf.fit_transform(df["contenido"])

            # Matriz de similitud coseno
            similaridad = cosine_similarity(matriz_tfidf)

            # Función recomendadora
            def recomendar_libros(titulo, n=3):
                if titulo not in df["title"].values:
                    return []
                idx = df[df["title"] == titulo].index[0]
                scores = list(enumerate(similaridad[idx]))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                indices = [i[0] for i in scores[1:n+1]]
                return df.iloc[indices][["title", "author", "genre", "description"]]

            # Interfaz de selección
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                libro_seleccionado = st.selectbox("📘 Selecciona un libro:", df["title"].values)
            with col2:
                n_recomendaciones = st.slider("N° de recomendaciones", 1, 5, 3)

            if st.button("✨ Recomendar"):
                recomendaciones = recomendar_libros(libro_seleccionado, n=n_recomendaciones)
                if len(recomendaciones) == 0:
                    st.warning("No se encontraron libros similares.")
                else:
                    st.markdown(f"### 📚 Libros similares a **{libro_seleccionado}**")
                    st.markdown("---")
                    for _, row in recomendaciones.iterrows():
                        st.markdown(f"""
                        <div class="book-card">
                            <div class="title">{row['title']}</div>
                            <div class="author">por {row['author']}</div>
                            <div class="genre">{row['genre']}</div>
                            <p>{row['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Error al leer el archivo: {e}")
else:
    st.warning("📥 Esperando a que subas tu archivo CSV...")
