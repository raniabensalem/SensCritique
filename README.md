                +-------------------+
                |   Base de données  |
                | critiques (films)  |
                +-------------------+
                         |
                         v
                 +-------------------+
                 |   Prétraitement    |
                 | - Nettoyage texte  |
                 | - Stopwords FR     |
                 | - TF-IDF           |
                 +-------------------+
                         |
                         v
                +--------------------+
                |  Similarité Cosinus |
                |  (matrice NxN)      |
                +--------------------+
                         |
                         v
                +----------------------+
                | API / Service Python |
                | Entrée : critique_id |
                | Sortie : top-k       |
                | critiques similaires |
                +----------------------+
                         |
                         v
              +-------------------------+
              |    Interface utilisateur |
              |   (SensCritique Front)   |
              | => suggestions affichées |
              +-------------------------+
