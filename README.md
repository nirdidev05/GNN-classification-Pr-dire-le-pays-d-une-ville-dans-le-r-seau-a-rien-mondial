lien

# Prédire le pays d'une ville dans le réseau aérien mondial (GNN)

Projet du module **BIML-GNN** (M2 IA, Université Claude Bernard Lyon 1).
Trinôme : **BENBOUTA Walid** (p2513426), **BOUMAZA Choayb** (p2513241), **BENHAMAID Mohamed Amine** (p2512882).

> **Projet réalisé sur Kaggle.** Tout le travail a été fait dans un notebook Kaggle, avec un GPU NVIDIA T4 et le graphe chargé comme dataset Kaggle. Le notebook de ce dépôt est l'export de celui-ci, avec les sorties de notre dernière exécution. La section 7 explique comment le relancer sur Kaggle ou sur une autre machine.

**Notebook Kaggle :** [https://www.kaggle.com/code/benboutawalid/notebook9ef703683a](https://www.kaggle.com/code/benboutawalid/notebook9ef703683a)

## 1. Le sujet en bref

Nous avons un graphe du réseau aérien mondial :

- un **nœud** = une ville avec un aéroport (3 363 villes) ;
- une **arête** = une ligne aérienne directe entre deux villes (13 547 lignes).

Chaque ville a une latitude, une longitude, une population et un pays.

**But (tâche T1) :** deviner le **pays de chaque ville**. C'est de la **classification de nœuds**.

On est dans le cadre vu en cours (slides 14 et 35) : **transductif et semi-supervisé**. On voit tout le graphe, mais on ne connaît le pays que des villes d'entraînement. Le modèle doit deviner les autres.

**Résultat final :**

| Modèle                                                             | F1-macro                 | Accuracy |
| ------------------------------------------------------------------- | ------------------------ | -------- |
| **GAT pondéré sur Node2Vec** (notre modèle, exécution V2) | **0,851 ± 0,021** | 0,890    |
| Même modèle, exécution V1                                        | 0,859                    | 0,892    |
| Label Propagation (meilleure baseline)                              | 0,831 ± 0,015           | 0,898    |

Notre modèle est le seul qui bat la Label Propagation sur les 5 découpages (plis) de test, et ce dans nos deux exécutions complètes.

---

## 2. Contenu du dépôt

| Fichier                                     | Rôle                                                                             |
| ------------------------------------------- | --------------------------------------------------------------------------------- |
| `gnn_prediction_pays_reseau_aerien.ipynb` | Tout le code, avec les sorties de notre dernière exécution (V2, 8 octobre 2026) |
| `README.md`                               | Ce fichier                                                                        |

Le notebook se lit dans l'ordre :

| Section | Ce que l'on y fait                                                                             |
| ------- | ---------------------------------------------------------------------------------------------- |
| 0       | Installation des bibliothèques, imports, graine, cellule de reprise (facultative)             |
| 1       | Exploration du graphe : degrés, composantes, carte, pays                                      |
| 2       | Nettoyage des pays, classe`OTHER`, calcul des features                                       |
| 3       | Protocole : découpage en 5 plis, validation, métriques                                       |
| 4       | Baselines : classe majoritaire, MLP, k-NN, Label Propagation, Node2Vec                         |
| 5       | Premiers essais de GNN (phase de recherche ; ses chiffres ne sont pas comparables entre eux)   |
| 6       | **Comparaison finale** : tous les modèles, tous les scénarios, mêmes réglages        |
| 7       | **Améliorations** : arêtes géographiques, grille plus large, pondération des classes |

Les résultats à retenir sont ceux des **sections 4, 6 et 7**.

---

## 3. Les difficultés du problème, et ce qu'on a décidé

| Ce qu'on a observé                                                                                                   | Ce qu'on a décidé                                                                                                                                                                                                                |
| --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Les pays sont très déséquilibrés : les USA ont 650 villes, et 58 pays sur 99 en ont moins de 20.                  | On juge les modèles avec le**F1-macro** : chaque pays compte autant. L'accuracy compte chaque ville, donc elle favorise les gros pays : un modèle qui répond toujours « USA » a 19 % d'accuracy mais 0,3 % de F1-macro. |
| 62 pays n'ont qu'une seule ville.                                                                                     | Les pays de moins de 5 villes sont regroupés dans une classe`OTHER`, ce qui donne **100 classes**. Il faut au moins 5 villes par pays pour en avoir une dans chacun des 5 plis.                                           |
| La latitude et la longitude donnent presque la réponse : la ville la plus proche est presque toujours du même pays. | On fait deux familles d'expériences :**A** (avec les coordonnées) et **B** (sans les coordonnées, pour voir ce que le graphe apporte seul).                                                                         |
| La population vaut 10 000 pour la moitié des villes (valeur par défaut).                                            | On ne l'utilise pas.                                                                                                                                                                                                               |
| 170 villes n'ont**aucun** voisin de leur pays : tous leurs vols partent à l'étranger.                         | On les surveille dans les diagnostics. Ce sont elles qui piègent les modèles qui « copient » leurs voisins.                                                                                                                    |

---

## 4. Les features et les scénarios

### 4.1 Les features

Aucune feature n'utilise les pays. On peut donc les calculer sur tout le graphe sans tricher.

| Feature      | Taille | Ce que c'est                                                                               | Pourquoi                                                                                                                                                                                    |
| ------------ | ------ | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Coordonnées | 3      | Latitude et longitude transformées en un point sur une sphère (x, y, z)                  | Pas de saut entre −180° et +180°                                                                                                                                                         |
| Structure    | 4      | log(degré), log(PageRank), clustering, closeness                                          | Décrire le**rôle** de la ville (gros hub ou petit aéroport régional)                                                                                                              |
| Distances    | 3      | Longueur moyenne et minimale des vols de la ville, part des vols de moins de 500 km        | Savoir si les voisins sont proches, donc s'ils sont probablement du même pays. Les GNN du cours n'utilisent pas d'attributs sur les arêtes, donc on met cette information sur les nœuds. |
| Node2Vec     | 128    | Un vecteur par ville, appris par marches aléatoires sur le graphe,**sans les pays** | Donner une « position dans le graphe » à chaque ville (méthode citée slide 31)                                                                                                         |
| VGAE         | 128    | Un vecteur par ville, appris par l'auto-encodeur du cours (slides 24 à 27)                | Comparer Node2Vec à une méthode vue en cours                                                                                                                                              |
| Identité    | —     | Une ligne apprise par ville (`nn.Embedding`), ce qui revient à X = matrice identité    | Tester si un GNN réussit quand chaque ville est reconnaissable                                                                                                                             |

### 4.2 Les scénarios

Le graphe est toujours celui des vols ; seule l'entrée X change. Chaque scénario répond à une question.

| Scénario  | Entrée                      | Question                                                                                |
| ---------- | ---------------------------- | --------------------------------------------------------------------------------------- |
| A          | coordonnées + structure (7) | Le graphe aide-t-il quand on a déjà les coordonnées ?                                |
| A+dist     | A + distances (10)           | La longueur des vols aide-t-elle ?                                                      |
| A+dist+n2v | A+dist + Node2Vec (138)      | Que donne le modèle avec toutes les informations ?                                     |
| B          | structure seule (4)          | Le graphe suffit-il sans la position ?                                                  |
| B+id       | identité apprise            | Un GNN fait-il aussi bien que la Label Propagation si chaque ville est reconnaissable ? |
| B+n2v      | Node2Vec seul (128)          | Le GNN améliore-t-il un bon embedding ?                                                |
| B+vgae     | VGAE seul (128)              | Le VGAE du cours vaut-il Node2Vec ?                                                     |

**Pourquoi un GNN échoue en B (lien avec la slide 11).** Pour un GNN, l'ordre des nœuds n'a pas de sens ; c'est l'équivariance par permutation. Deux villes qui ont le même voisinage et les mêmes features reçoivent donc forcément la même prédiction, même si elles sont dans deux pays différents. La Label Propagation et Node2Vec n'ont pas ce problème, car ils donnent à chaque ville quelque chose qui lui est propre (son label propagé, ou un vecteur appris rien que pour elle). La preuve :

| Même GCN, entrée différente | F1-macro |
| ------------------------------ | -------- |
| B (structure seule)            | 0,359    |
| B+id (identité apprise)       | 0,803    |
| B+n2v (Node2Vec)               | 0,832    |

---

## 5. Les modèles et leur architecture

### 5.1 Les baselines

- **Classe majoritaire** : répond toujours « USA ».
- **MLP** : un réseau de neurones classique qui **ne regarde pas les arêtes**.
- **k-NN sur les coordonnées** : vote des villes d'entraînement les plus proches.
- **Label Propagation (LP)** : les pays connus « coulent » le long des arêtes. C'est la même matrice que le GCN de la slide 13, mais appliquée directement aux labels, sans poids appris.
- **Node2Vec + k-NN** : on apprend un vecteur par ville, puis un k-NN décide.

### 5.2 Les GNN : un schéma commun

Tous les GNN ont la même forme :

```
X (features ou identité)
  -> [dropout -> couche de graphe -> activation]   répété L-1 fois
  -> dropout -> couche de graphe -> 100 sorties (une par classe)
  -> softmax -> pays prédit
```

Seule la **couche de graphe** change d'un modèle à l'autre.

| Modèle               | Ce que fait la couche                                                                                                      | Slides                                       | Comment la ville elle-même est prise en compte  |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------ |
| **GCN**         | moyenne pondérée de la ville et de ses voisins, poids fixés par les degrés : 1/√(dᵢ·dⱼ)                            | 12, 13, 15                                   | Elle n'est qu'**un voisin parmi d'autres** |
| **GraphSAGE**   | W₁·(la ville) + W₂·(moyenne des voisins)                                                                               | 35                                           | Elle a**sa propre matrice** W₁            |
| **GAT (GATv2)** | somme des voisins, chacun avec un poids**appris** par l'attention (softmax) ; 8 têtes concaténées, activation ELU | 19 à 21, code du cours                      | Elle reçoit elle aussi un poids appris          |
| **SIGN**        | un MLP sur [X, ŜX, Ŝ²X, Ŝ³X], où la propagation est calculée une fois à l'avance                                   | idée tirée de la slide 13 (associativité) | Le bloc X reste intact                           |
| **MLP**         | même tête, sans aucune arête                                                                                            | 33                                           | Modèle de contrôle                             |

**Pourquoi c'est important.** Sur les 170 villes dont tous les vols partent à l'étranger, plus un modèle « protège » l'information de la ville elle-même, mieux il s'en sort :

| Modèle                                   | MLP   | SIGN  | GraphSAGE | GAT   | GCN   | LP    |
| ----------------------------------------- | ----- | ----- | --------- | ----- | ----- | ----- |
| Accuracy sur ces 170 villes (scénario A) | 0,618 | 0,500 | 0,347     | 0,153 | 0,071 | 0,024 |

Le GAT est le meilleur GNN dans tous les scénarios. Son attention a appris à faire **moins confiance aux voisins lointains** : pour 81 % des villes, plus le vol est long, plus le poids donné au voisin est faible.

### 5.3 La pondération des classes

Sans pondération, la loss fait la moyenne sur toutes les villes, donc les gros pays (USA, Canada…) dominent l'entraînement. On donne à chaque ville du pays c le poids :

```
w_c = N_train / (C × n_c)      (N_train : villes d'entraînement, C = 100 classes, n_c : villes du pays c)
```

Chaque pays pèse alors autant dans la loss, exactement comme dans le F1-macro. Dans le code, c'est l'argument `weight=` de `F.cross_entropy`. Cette pondération n'est pas dans les slides, mais elle ne change que la loss, pas le modèle.

---

## 6. Le protocole

- **Validation croisée** : 5 plis stratifiés (`StratifiedKFold`, `random_state=42`). Dans chaque pli, 673 villes sont en test, 2 353 en entraînement et 337 en validation.
- **Le test ne sert jamais à choisir.** Les réglages (hyperparamètres) et le moment où on arrête l'entraînement (early stopping, patience 50) sont choisis sur le **F1-macro de validation**.
- **Même grille pour tous les GNN** :
  - largeur {64, 128, 256} ;
  - dropout {0,1 ; 0,3 ; 0,5} ;
  - nombre de couches {2, 3} ;
  - optimiseur Adam, weight decay 5·10⁻⁴.
- **Résultats** donnés en moyenne ± écart-type sur les 5 plis.
- **Arêtes géographiques** : on a testé l'ajout d'arêtes vers les k villes les plus proches. La validation a choisi k = 0 (pas d'arêtes géographiques) : dès k = 1, le F1 baisse.

---

## 7. Reproduire nos résultats

Le notebook a été écrit et exécuté **sur Kaggle**. C'est donc la façon la plus simple de le relancer (7.3). Il tourne aussi sur une autre machine (ordinateur personnel, serveur, Google Colab) en changeant une seule ligne (7.4).

### 7.1 Données

- Fichier : `airportsAndCoordAndPop.graphml` (le graphe fourni pour le TP).
- Le notebook lit ce fichier à partir de la variable `RAW_PATH`, définie dans la **cellule des imports** (3ᵉ cellule de code) :

  ```python
  RAW_PATH = Path("/kaggle/input/datasets/benboutawalid/graphe/airportsAndCoordAndPop.graphml")
  ```
- C'est le **seul chemin à adapter**. Tous les autres dossiers (`data/processed/`, `results/`, `results/figures/`) sont relatifs et créés automatiquement.

### 7.2 Environnement utilisé pour nos résultats

| Élément         | Version (exécution V2, Kaggle)                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| Plateforme        | Kaggle Notebooks,**GPU NVIDIA T4**                                                         |
| Python            | 3.13 (d'après les métadonnées du notebook)                                                    |
| PyTorch           | 2.11 (CUDA 12.8)                                                                                 |
| PyTorch Geometric | 2.8.0.post1                                                                                      |
| pyg_lib           | 0.9.0 (sert aux marches aléatoires de Node2Vec)                                                 |
| Autres            | numpy, pandas, scikit-learn, scipy, networkx, matplotlib, seaborn (déjà installés sur Kaggle) |

### 7.3 Guide d'exécution sur Kaggle (notre environnement)

**Avant de commencer :** il faut un compte Kaggle **vérifié par téléphone**, sinon Kaggle ne donne accès ni au GPU ni à Internet. Internet est obligatoire pour les deux cellules d'installation (`pip`).

1. **Créer le notebook.** Sur kaggle.com : *Create → New Notebook*, puis *File → Import Notebook* et choisir le fichier `.ipynb` de ce dépôt.
2. **Ajouter les données.** Dans le panneau de droite : *Add Input*.
   - Si vous avez accès à notre dataset `benboutawalid/graphe`, ajoutez-le : le chemin de `RAW_PATH` est déjà le bon.
   - Sinon, créez votre propre dataset (*Create → New Dataset*, déposez `airportsAndCoordAndPop.graphml`), ajoutez-le au notebook, puis copiez le chemin du fichier affiché dans le panneau *Input* (bouton « copier le chemin ») et collez-le dans `RAW_PATH`.
3. **Régler la session.** Dans *Settings* (panneau de droite) :
   - *Accelerator* → **GPU T4** (×1 suffit) ;
   - *Internet* → **On**.
4. **Installer PyTorch Geometric.** Exécuter les **deux premières cellules de code** :
   - la 1ʳᵉ installe `torch_geometric` ;
   - la 2ᵉ lit la version de PyTorch et de CUDA de la session, puis installe le `pyg_lib` qui correspond (depuis `data.pyg.org`).
5. **Redémarrer le noyau** (*Run → Restart & clear cell outputs*), comme l'indique le message affiché par la 2ᵉ cellule. Ce redémarrage ne sert qu'en exécution cellule par cellule : avec *Save & Run All*, les installations ont lieu avant le premier import de `torch_geometric`, donc le notebook peut s'exécuter d'une traite.
6. **Lancer tout le notebook.**
   - *Run All* : exécution interactive. Il faut garder l'onglet ouvert pendant tout le calcul.
   - *Save Version → Save & Run All (Commit)* : **recommandé**. Le calcul tourne en arrière-plan sur les serveurs de Kaggle, et on peut fermer le navigateur. Les sorties et les fichiers créés apparaissent ensuite dans l'onglet *Output* de la version.
7. **Récupérer les résultats.** Dans l'onglet *Output* de la version, on peut télécharger `results/` (fichiers `.csv`, `.npz`, figures) et `data/processed/` (embeddings).

**Quota GPU.** Kaggle limite le temps de GPU par semaine. Une exécution complète prend environ 4 h, donc il faut vérifier qu'il reste assez de quota avant de lancer.

### 7.4 Guide d'exécution sur une autre machine

Le code ne dépend de Kaggle qu'à trois endroits :

| Endroit                                             | Sur Kaggle                                           | Sur une autre machine                                                                |
| --------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `RAW_PATH` (cellule des imports)                  | `/kaggle/input/...`                                | **À modifier** : chemin local du `.graphml`                                 |
| Cellule « REPRISE » et tableau récapitulatif 6.7 | Cherchent d'anciens résultats dans`/kaggle/input` | Ne font rien (le dossier n'existe pas) ; le notebook utilise`results/` normalement |
| Choix du GPU                                        | GPU T4                                               | Automatique :`DEVICE = cuda` si une carte NVIDIA est disponible, sinon `cpu`     |

**Étapes :**

1. **Récupérer le code et les données.**

   ```bash
   git clone https://github.com/nirdidev05/data_walid.git GNN-classification-pays-ville-reseau-aerien-mondial
   cd GNN-classification-pays-ville-reseau-aerien-mondial
   mkdir -p data/raw
   # copier airportsAndCoordAndPop.graphml dans data/raw/
   ```
2. **Créer un environnement Python** (Python 3.10 ou plus récent ; nous avons utilisé 3.13 sur Kaggle).

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows : .venv\Scripts\activate
   ```
3. **Installer PyTorch.** La commande dépend de la machine (GPU NVIDIA ou non). Le plus sûr est de la copier depuis le sélecteur de pytorch.org (*Get Started*). Par exemple :

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu      # sans GPU
   pip install torch --index-url https://download.pytorch.org/whl/cu128    # GPU NVIDIA, CUDA 12.8 (comme sur Kaggle)
   ```
4. **Installer les autres bibliothèques.**

   ```bash
   pip install torch_geometric numpy pandas scikit-learn scipy networkx matplotlib seaborn jupyter
   ```

   `pyg_lib` est installé par la 2ᵉ cellule du notebook, qui choisit automatiquement la bonne version. Si aucune version de `pyg_lib` n'existe pour votre système, installez `torch_cluster` depuis la même page (`https://data.pyg.org/whl/torch-<version>+<cpu ou cuXXX>.html`) : Node2Vec peut utiliser l'un ou l'autre pour ses marches aléatoires.
5. **Modifier `RAW_PATH`** dans la cellule des imports :

   ```python
   RAW_PATH = Path("data/raw/airportsAndCoordAndPop.graphml")
   ```
6. **Lancer Jupyter depuis le dossier du dépôt**, pour que `data/` et `results/` soient créés au bon endroit :

   ```bash
   jupyter notebook
   ```

   Exécuter les deux cellules d'installation, redémarrer le noyau, puis *Run All*. Si `torch_geometric` est déjà installé (étape 4), la 1ʳᵉ cellule ne fait rien de plus.

**Sur Google Colab :** même démarche. On importe le notebook, on choisit un GPU (*Exécution → Modifier le type d'exécution*), on dépose le `.graphml` dans `/content/` et on met `RAW_PATH = Path("/content/airportsAndCoordAndPop.graphml")`.

**Sans GPU :** le notebook fonctionne, mais les sections 6 et 7 (grille de 18 configurations × 5 plis pour chaque modèle et chaque scénario) seront nettement plus longues que sur le T4. Nous n'avons pas mesuré cette durée.

### 7.5 Durée et fichiers créés

**Durée totale : environ 4 h** sur GPU T4 (Kaggle).

| Partie               | Durée          |
| -------------------- | --------------- |
| Baselines + Node2Vec | environ 17 min  |
| Section 6            | environ 83 min  |
| Section 7            | environ 115 min |

**Fichiers créés :**

- dans `results/` : les scores pli par pli (`.csv`), les prédictions (`.npz`) et les figures (`results/figures/`) ;
- dans `data/processed/` : les embeddings Node2Vec et VGAE, gardés en cache.

### 7.6 Si l'exécution s'arrête en cours de route

Les résultats sont sauvegardés après chaque entraînement, donc on ne perd pas tout.

- **Sur Kaggle :** on ajoute la sortie de la version précédente comme entrée du notebook (*Add Input → Your Work / Notebook Output*). La cellule « REPRISE » copie alors les embeddings (`node2vec_*.pt`, `vgae_*.pt`) et les résultats de la section 6 (`final_raw_*.csv`, `final_preds_*.npz`) dans les dossiers locaux. Le notebook ne recalcule pas ce qui existe déjà.
- **Sur une autre machine :** rien à faire si on relance dans le même dossier, car les fichiers sont déjà dans `data/processed/` et `results/`. Pour reprendre une exécution faite ailleurs (par exemple sur Kaggle), on copie ces fichiers à la main dans les mêmes dossiers.

### 7.7 Ce qui peut changer d'une exécution à l'autre

- **Identique à chaque fois** : la Label Propagation, le k-NN, le MLP et SIGN (quand l'entrée ne contient pas Node2Vec).
- **Petites différences possibles** :
  - **GCN, GraphSAGE et GAT sur GPU** : certains calculs du GPU ne donnent pas exactement le même résultat d'une fois à l'autre. On a mesuré jusqu'à 0,015 d'écart sur la moyenne.
  - **Node2Vec** est réentraîné à chaque exécution, donc les scénarios qui l'utilisent peuvent bouger un peu. Pour retrouver exactement nos chiffres, il faut reprendre nos embeddings en cache.
- **Les conclusions restent les mêmes** dans nos deux exécutions complètes (V1 et V2) : le GAT est le meilleur GNN, l'embedding est l'ingrédient le plus important, et le GAT pondéré B+n2v bat la LP sur les 5 plis.

---

## 8. Résultats principaux (V2, F1-macro ± écart-type sur 5 plis)

| Méthode                | A                        | A+dist+n2v               | B                        | B+n2v                    |
| ----------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| k-NN sur coordonnées   | 0,773 ± 0,018           | —                       | —                       | —                       |
| Node2Vec + k-NN         | 0,781 ± 0,024           | —                       | 0,781 ± 0,024           | —                       |
| Label Propagation       | **0,831 ± 0,015** | —                       | **0,831 ± 0,015** | —                       |
| MLP                     | 0,629 ± 0,049           | 0,814 ± 0,028           | 0,158 ± 0,027           | 0,787 ± 0,030           |
| SIGN                    | 0,711 ± 0,028           | 0,833 ± 0,007           | 0,381 ± 0,030           | 0,815 ± 0,029           |
| GCN                     | 0,640 ± 0,061           | 0,827 ± 0,034           | 0,359 ± 0,043           | 0,832 ± 0,020           |
| GraphSAGE               | 0,668 ± 0,045           | 0,840 ± 0,015           | 0,483 ± 0,051           | 0,815 ± 0,019           |
| GAT                     | 0,768 ± 0,018           | 0,851 ± 0,024           | 0,729 ± 0,030           | 0,845 ± 0,024           |
| **GAT pondéré** | —                       | **0,859 ± 0,034** | —                       | **0,851 ± 0,021** |

Le tableau complet (7 scénarios, accuracy, étude d'ablation) est dans le rapport.

---

## 9. Utilisation d'une IA générative

**Modèle utilisé :** Claude, d'Anthropic, dans l'application Claude. Le modèle configuré était **Claude Opus 5.5** (identifiant `claude-opus-5-5`).

### Pourquoi on l'a utilisée

- lire et analyser les résultats ;
- nous expliquer les notions du cours, et les méthodes trouvées sur les sites et dans les articles de la bibliographie ;
- discuter les choix de protocole et de modèles ;
- debugging ;
- analyser les sorties du notebook ;
- écrire le code LaTeX des rapports, suivant nos demandes.

### Comment on l'a utilisée

- **Lire et analyser les résultats, analyser les sorties.** On lui donnait le notebook avec ses sorties, et elle commentait les tableaux : comparaisons pli par pli, villes mal classées, rôle de chaque feature. Par exemple, elle nous a aidés à comprendre pourquoi un GNN échoue sans les coordonnées (équivariance par permutation). On a discuté chaque explication. Plusieurs de ses prédictions se sont révélées fausses une fois le code exécuté, et on l'a noté dans le rapport détaillé.
- **Explications du cours et des méthodes.** Elle nous a réexpliqué les notions des slides (GCN, GAT, GraphSAGE, VGAE, message passing, cadre transductif) et des méthodes vues dans les articles cités (Label Propagation, Node2Vec, SIGN, pondération des classes), pour qu'on puisse justifier nos choix.
- **Discuter les choix de protocole et de modèles.** Choix du F1-macro, sélection sur la validation, même grille pour tous les modèles, scénarios A et B, pondération des classes, et pistes d'amélioration de la section 7. Les décisions finales ont été prises par nous.
- **Debugging.**
  - Cellules qui dépendaient les unes des autres dans le mauvais ordre, variables non définies (`PATIENCE`, `SCENARIOS`).
  - Absence de graine (seed).
  - Node2Vec pas assez entraîné, et fichier de cache mal nommé.
  - Erreurs qui arrêtaient *Save & Run All* sur Kaggle.
- **Code des cellules.** Elle a proposé des cellules (évaluation en F1-macro, scénarios, VGAE, protocole commun, diagnostics, section 7). Nous les avons placées, exécutées et vérifiées nous-mêmes dans le notebook. **L'IA n'a jamais exécuté le code sur nos données : tous les chiffres viennent de nos exécutions.**
- **Code LaTeX.** Elle a écrit le code LaTeX du rapport (gabarit LNCS) et d'un rapport détaillé pour réviser l'écrit, à partir de nos résultats et selon nos demandes.

Notre façon de travailler : l'IA propose des cellules complètes, on les place et on les exécute, puis on discute ensemble des résultats.

### Exemples de prompts utilisés

> « N'écris pas de code pour l'instant. Quand tu me donnes du code, tu ne l'exécutes pas : tu me donnes les cellules à ajouter, je te donne les résultats et on en débat. »

> « Voici le notebook : analyse le code et les sorties en détail, ne survole aucune cellule, et analyse tous les résultats en détail. »

> « Je veux aussi améliorer le cas B, parce que je pense qu'il est sévère pour la partie GNN et qu'il teste vraiment mes modèles, non ? Il faut qu'on modifie les features. »

> « En suivant ces fichiers explicatifs, fais le rapport de notre travail […] et écris un rapport détaillé qui suit strictement les consignes. »

**Responsabilité.** Les choix finaux (scénarios, modèles, métrique, modèle retenu) et la vérification des résultats sont faits par le trinôme.
