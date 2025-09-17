
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re

FIGHT_CSV = "fight_club_critiques.csv"
INTER_CSV = "interstellar_critique.csv"
KNOWN_TEXT_COLS = ["text","critique","review","content","comment","body","commentaire"]

def clean_html(s):
    if s is None:
        return ""
    s = str(s)
    try:
        s = BeautifulSoup(s, "html.parser").get_text(separator=" ")
    except Exception:
        s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def detect_text_column(df):
    for c in df.columns:
        if c.lower() in KNOWN_TEXT_COLS:
            return c
    candidates = []
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            avg_len = df[c].dropna().astype(str).map(len).mean() if df[c].dropna().shape[0]>0 else 0
            candidates.append((c, avg_len))
    if not candidates:
        return df.columns[0]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def build_dataframe():
    df1 = pd.read_csv(FIGHT_CSV, dtype=object)
    df2 = pd.read_csv(INTER_CSV, dtype=object)
    df1["film"] = "Fight Club"
    df2["film"] = "Interstellar"
    df = pd.concat([df1, df2], ignore_index=True)
    if "critique_id" not in df.columns:
        df = df.reset_index().rename(columns={"index":"critique_id"})
    return df

def prepare_text_column(df):
    text_col = detect_text_column(df)
    df["cleaned_text"] = df[text_col].fillna("").astype(str).apply(clean_html)
    return df, text_col

def build_tfidf(df):
    vec = TfidfVectorizer(stop_words="french", ngram_range=(1,2), max_df=0.85)
    tfidf = vec.fit_transform(df["cleaned_text"])
    return vec, tfidf

def recommend(df, cosine_sim, index, topk=3):
    if index < 0 or index >= len(df):
        raise IndexError("index hors limites")
    film = df.loc[index, "film"]
    same_mask = df["film"] == film
    candidates = [i for i,flag in enumerate(same_mask) if flag and i != index]
    if not candidates:
        return []
    sims = [(i, float(cosine_sim[index, i])) for i in candidates]
    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:topk]
    results = []
    for i,score in top:
        results.append({
            "index": int(i),
            "critique_id": df.loc[i,"critique_id"] if "critique_id" in df.columns else None,
            "film": film,
            "score": score,
            "preview": df.loc[i, "cleaned_text"][:400]
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Index (0-based) de la critique à utiliser comme requête")
    parser.add_argument("--topk", type=int, default=3, help="Nombre de recommandations")
    args = parser.parse_args()

    print("Chargement des CSVs")
    df = build_dataframe()
    df, text_col = prepare_text_column(df)
    print(f"Colonne texte détectée: '{text_col}'")
    print("Construction TF-IDF")
    _, tfidf = build_tfidf(df)
    print("Calcul matrice similarité")
    cosine_sim = cosine_similarity(tfidf, tfidf)

    qidx = args.index
    print("\nCritique source (preview):")
    print(df.loc[qidx, "cleaned_text"][:500])
    print("\nTop recommandations (même film):")
    recs = recommend(df, cosine_sim, qidx, topk=args.topk)
    if not recs:
        print("Aucune suggestion (pas assez de critiques pour le film ou texte vide).")
        return
    for i,r in enumerate(recs, start=1):
        print(f"\n#{i} | index={r['index']} | score={r['score']:.4f}")
        print(r["preview"][:400])

if __name__ == "__main__":
    main()
