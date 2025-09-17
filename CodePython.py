import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Chargement des datasets
# Les deux fichiers CSV contiennent les critiques
fight = pd.read_csv("fight_club_critiques.csv")
inter = pd.read_csv("interstellar_critique.csv")

# On ajoute une colonne "film" pour savoir à quel film appartient la critique
fight["film"] = "Fight Club"
inter["film"] = "Interstellar"

# Fusionner les deux datasets en un seul
df = pd.concat([fight, inter], ignore_index=True)

# Nettoyage : remplacer les valeurs manquantes dans la colonne des critiques
df["review_content"] = df["review_content"].fillna("")

# 2. Transformation TF-IDF
# On convertit les critiques en vecteurs numériques
vectorizer = TfidfVectorizer(stop_words="french")
tfidf_matrix = vectorizer.fit_transform(df["review_content"])

# Calculer la similarité cosinus entre toutes les critiques
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3. Fonction de recommandation
def recommander(index, topk=3):
    """
    Trouver les critiques similaires à une critique donnée (par son index).
    On ne recommande que des critiques du même film.
    """
    film = df.loc[index, "film"]
    similarities = list(enumerate(cosine_sim[index]))

    # Garder seulement les critiques du même film et ignorer la critique elle-même
    similarities = [(i, score) for i, score in similarities if df.loc[i, "film"] == film and i != index]

    # Trier par score décroissant
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:topk]

    # Construire les résultats
    resultats = []
    for i, score in similarities:
        resultats.append({
            "film": df.loc[i, "film"],
            "score": round(score, 3),
            "review": df.loc[i, "review_content"][:250]  # aperçu des 250 premiers caractères
        })
    return resultats
    
# 4. Exemple d’utilisation
if __name__ == "__main__":
    # Choisir un index de critique pour tester (ici la première critique)
    index_test = 0

    print("=== Critique de base ===")
    print(df.loc[index_test, "review_content"], "\n")

    print("=== Critiques similaires ===")
    recommandations = recommander(index_test, topk=3)
    for r in recommandations:
        print(f"[{r['film']}] (score={r['score']})")
        print(r["review"], "\n")
