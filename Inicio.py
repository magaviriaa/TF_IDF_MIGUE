import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ---------------------------
# LYRICS LAB — Taylor's Version
# (misma estructura, nuevo copy)
# ---------------------------

st.title("Lyrics Lab — TF-IDF (Taylor’s Version)")

st.write("""
Cada línea que pegues se toma como un **documento** (verso/fragmento).  
⚠️ El análisis está configurado para **inglés** (stemming y stopwords en inglés).

**¿Qué hace esto?** Calcula TF-IDF sobre tus versos y encuentra cuál es más relevante para tu **pregunta**.
Ideal para jugar con letras y detectar en qué verso se responde mejor a tu duda.
""")

# Ejemplo inicial en inglés (versos tipo demo)
text_input = st.text_area(
    "Pega tus versos o líneas (uno por línea, en inglés):",
    "We dance in the kitchen at midnight.\n"
    "The city lights shimmer on the river.\n"
    "You said forever under violet skies."
)

question = st.text_input("Escribe tu pregunta (en inglés):", "Who is dancing?")

# Inicializar stemmer para inglés
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    # minúsculas
    text = text.lower()
    # solo letras y espacios
    text = re.sub(r'[^a-z\\s]', ' ', text)
    # tokens de longitud > 1
    tokens = [t for t in text.split() if len(t) > 1]
    # stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("Calcular TF-IDF y buscar respuesta"):
    documents = [d.strip() for d in text_input.split("\\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento (verso).")
    else:
        # Vectorizador con nuestro tokenizer (usa stemming)
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None  # necesario cuando definimos tokenizer
        )

        # Ajuste con documentos (versos)
        X = vectorizer.fit_transform(documents)

        # Matriz TF-IDF
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Line {i+1}" for i in range(len(documents))]
        )

        st.write("### Matriz TF-IDF (stems)")
        st.dataframe(df_tfidf.round(3))

        # Vector de la pregunta
        question_vec = vectorizer.transform([question])

        # Similitud coseno
        similarities = cosine_similarity(question_vec, X).flatten()

        # Documento más parecido
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.write("### Pregunta y verso más relevante")
        st.write(f"**Pregunta:** {question}")
        st.write(f"**Verso destacado (Line {best_idx+1}):** {best_doc}")
        st.write(f"**Puntaje de similitud:** {best_score:.3f}")

        # Todas las similitudes
        sim_df = pd.DataFrame({
            "Line": [f"Line {i+1}" for i in range(len(documents))],
            "Text": documents,
            "Similarity": similarities
        })
        st.write("### Puntajes de similitud (ordenados)")
        st.dataframe(sim_df.sort_values("Similarity", ascending=False))

        # Stems de la pregunta presentes en el verso elegido
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        # columnas del df_tfidf son los stems del vocab
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        st.write("### Stems de la pregunta presentes en el verso elegido:", matched)
