Système de recommandation de critiques
Contexte
SensCritique souhaite permettre aux utilisateurs de voir des critiques similaires lorsqu’ils lisent une critique d’un film.  
Exemple :  
- Si je lis une critique sur *Fight Club* qui dit qu’il y a trop de bagarres, le système doit proposer d’autres critiques du même film exprimant une opinion proche.
Ce projet implémente cette fonctionnalité à l’aide de TF-IDF et de la similarité cosinus.

System Design : 
    A[Utilisateur lit une critique] --> B[Extraire le texte de la critique]
    B --> C[TF-IDF Vectorizer<br>(transforme le texte en vecteurs numériques)]
    C --> D[Calcul Similarité Cosinus<br>(entre toutes les critiques)]
    D --> E[Filtrer uniquement les critiques<br>du même film]
    E --> F[Recommandations<br>de critiques similaires]
