# Molecules Clustering

Clustering de dynamique moléculaire sur GPU (CUDA).

Le programme calcule la matrice de RMSD (Root Mean Square Deviation) entre toutes les paires de photographies d'une trajectoire moléculaire, puis effectue un clustering hiérarchique sur GPU pour regrouper les conformations similaires.

## Auteurs

- Florian DE BONI
- Paul NOLLET
- Yassine OUCHNA

---

## Structure du dépôt

```
molecules-clustering/
├── datasets/                  # À créer — contient les trajectoires brutes (.xtc, .pdb, etc.)
├── output/                    # Généré automatiquement — contient les fichiers .bin produits par md_reader
├── md_reader.cpp              # Étape 1 : lecture des trajectoires et conversion en binaire
├── main.cu                    # Étape 2 : programme principal CUDA (calcul RMSD + clustering)
├── gpu.cu / gpu.cuh           # Kernels CUDA : calcul des valeurs propres, matrice de distance
├── utils.cu / utils.cuh       # Utilitaires CPU/GPU : indexation triangle supérieur, chunking
├── FileUtils.cpp / .hpp       # Lecture/écriture des fichiers .bin (photographies binaires)
├── CudaTimer.cuh              # Utilitaire de mesure de temps via événements CUDA
└── Makefile                   # Système de compilation complet
```

### Rôle de chaque fichier

| Fichier | Rôle |
|---|---|
| `md_reader.cpp` | Lit les fichiers de trajectoire moléculaire dans `datasets/` via la bibliothèque **chemfiles**, et sérialise les coordonnées atomiques dans un fichier `.bin` dans `output/` |
| `main.cu` | Point d'entrée du programme GPU : charge le `.bin`, orchestre le calcul de la matrice RMSD par chunks, et effectue le clustering |
| `gpu.cu` / `gpu.cuh` | Kernels CUDA pour le calcul des valeurs propres (méthode de Cardano), calcul du RMSD par superposition, et lookup de la matrice triangulaire sur GPU |
| `utils.cu` / `utils.cuh` | Fonctions d'indexation du triangle supérieur, gestion du découpage en chunks, utilitaires partagés CPU/GPU |
| `FileUtils.cpp` / `.hpp` | Classe `FileUtils` : lecture rapide des photographies depuis le `.bin` (accès aléatoire, lecture en place) |
| `CudaTimer.cuh` | Minuterie CUDA basée sur `cudaEvent_t` pour profiler chaque étape du pipeline |

---

## Prérequis

- `g++` (C++17)
- `nvcc` (CUDA Toolkit)
- `cmake` (pour compiler chemfiles)
- `git`

Installation de cmake sur Linux :

```bash
sudo apt install cmake
```

L'architecture GPU cible est à décommenter dans le `Makefile` selon votre matériel :

```makefile
# CUDA_TARGET_FLAGS = -arch=sm_61   # GTX 1080
# CUDA_TARGET_FLAGS = -arch=sm_75   # RTX 2080-Ti
# CUDA_TARGET_FLAGS = -arch=sm_86   # RTX 3080
```

---

## Installation et compilation

### 1. Cloner le dépôt et préparer les données

```bash
git clone <url-du-repo>
cd molecules-clustering
mkdir datasets
```

Télécharger un jeu de données et le placer dans `datasets/` :

- **Petit dataset (~50 000 photographies)** — recommandé pour tester en local :  
  [1k5n_A_protein.zip](https://mdrepo.org/api/v1/get_download_instance/9280.zip) → décompresser dans `datasets/` ou éventuellement adapter le chemin directement dans `md_reader.cpp`.

- **Dataset complet (~200 000 photographies)** :  
  Décompresser dans `datasets/` dans plusieurs sous-dossiers.

---

### 2. Étape 1 — Compiler et exécuter `md_reader`

`md_reader` lit les fichiers de trajectoire dans `datasets/` et produit un fichier binaire `output/snapshots_coords_0.bin` consommé ensuite par le programme principal.

```bash
# Compiler md_reader (nécessite chemfiles — voir étape 2a)
g++ -o md_reader md_reader.cpp -DDP -I. \
    -I./chemfiles/include -I./chemfiles/build/include \
    -L./chemfiles/build -lchemfiles -O3

# Exécuter la conversion
./md_reader

# Le fichier output/snapshots_coords_0.bin est maintenant prêt
rm md_reader
```

> **Note :** La bibliothèque chemfiles doit être compilée au préalable (voir ci-dessous).

#### 2a. Compiler chemfiles

```bash
git clone https://github.com/chemfiles/chemfiles
mkdir chemfiles/build
cmake -S chemfiles -B chemfiles/build
cmake --build chemfiles/build --target chemfiles
```

---

### 3. Étape 2 — Compiler et exécuter le programme principal

```bash
# Compiler tous les fichiers CUDA et C++
nvcc -o main main.cu gpu.cu utils.cu objects/FileUtils.o \
     -DDP -I. -I./chemfiles/include -I./chemfiles/build/include \
     -L./chemfiles/build -lchemfiles \
     -L/usr/local/cuda/lib64 -lcudart -O3

# Exécuter sur le fichier binaire généré
./main output/snapshots_coords_0.bin
```

---

### 4. Compilation complète via `make` (recommandé)

Le `Makefile` automatise l'ensemble du pipeline (chemfiles → md_reader → main) :

```bash
# Compilation et exécution complète (dataset par défaut)
make

# Nettoyage des fichiers compilés
make clean
```
