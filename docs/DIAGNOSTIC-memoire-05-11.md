# Diagnostic mémoire et vitesse — 5 novembre 2026

Objet : comprendre pourquoi le run toy 2 000 pas est mort au pas 1502 (commit Windows à
12 591 Mo pour un graphe de ~5 800 arêtes et 587 nœuds), et pourquoi le débit réel est quatre
à huit fois plus lent que le chiffre d'acceptation de la v4 (0,283 s/pas).

Méthode : instrumentation de `mycelia/train.py` (RAM résidente, pic, commit, VRAM allouée et
réservée, à chaque rapport) et nouveau banc `scripts/bench_step.py` qui chronomètre séparément
le forward, le forward + backward et le pas complet, sur le même câblage.

Étiquettes : **(mesuré)**, **(estimé)**, **(supposé)**.

---

## 1. Instrumentation ajoutée

| Ce qui est mesuré | Où | Comment |
|---|---|---|
| `ram_mb` (working set), `ram_peak_mb`, `commit_mb` | chaque ligne de signes vitaux + `history.json` | `process_memory()` dans `train.py`, `ctypes` + `psapi.GetProcessMemoryInfo` |
| `gpu_alloc_mb` / `gpu_reserved_mb` | idem, seulement si le device est CUDA | `torch.cuda.memory_allocated()` / `memory_reserved()` |
| Effet d'un `gc.collect()` | ligne dédiée à chaque rapport | mesure avant/après, objets collectés |
| Coût par étage du pas | `scripts/bench_step.py` | médiane de 3 à 5 itérations, processus neuf pour chaque mesure mémoire |

`ctypes` et non `psutil` : `psutil` n'est pas une dépendance du projet, et le diagnostic doit
tourner sur une machine dont la mémoire de validation est déjà épuisée. Détail qui a coûté un
aller-retour : sans `argtypes` déclarés, `psapi.GetProcessMemoryInfo` échoue silencieusement
(le pseudo-handle 64 bits est tronqué) et renvoie des zéros.

---

## 2. Run d'observation — 400 pas, batch 16, séquence 10 (mesuré)

Commande : `python mycelia/train.py --steps 400 --batch-size 16 --report-every 50`
Log : `logs_diag_ram_400.txt` — 400 pas en 806 s, sortie propre, verrou relâché, `0 -> 2 -> 1`.

| Pas | Arêtes | Nœuds vivants | RAM (Mo) | Pic (Mo) | Commit (Mo) | ms/pas (moy. 20) |
|---|---|---|---|---|---|---|
| 50 | 400 | 514 | 1 690 | 1 710 | 3 735 | 1 130 |
| 100 | 792 | 514 | 1 695 | 1 710 | 4 367 | 1 290 |
| 150 | 1 157 | 514 | 1 697 | 1 710 | 4 947 | 1 323 |
| 200 | 1 544 | 514 | 1 700 | 1 710 | 5 540 | 1 499 |
| 250 | 1 797 | 514 | 1 705 | 1 710 | 5 884 | 2 078 |
| 300 | 2 178 | 514 | 1 709 | 1 710 | 6 424 | 2 614 |
| 350 | 2 351 | 486 | 1 735 | 1 738 | 6 643 | 3 445 |
| 400 | 2 734 | 499 | 1 716 | 1 738 | 7 146 | 3 616 |

**Les trois courbes mises en regard :**

* **RAM : plate.** 1 690 → 1 716 Mo, soit **+26 Mo pour +2 334 arêtes**. Le pic ne bouge pas
  (1 710 → 1 738). Il n'y a **ni fuite, ni rétention d'objets Python** : `gc.collect()` rend
  **0 Mo** à chaque rapport (46 à 55 objets collectés, tous déjà morts par comptage de
  références).
* **Arêtes : +5,7 par pas**, élagage compris — croissance attendue du câblage.
* **ms/pas : ×3,2** sur la même plage (1 130 → 3 616). Le pas 350 inclut le checkpoint
  d'horloge, mais le pas 300 (2 614 ms) et le pas 400 (3 616 ms) suffisent : le coût suit les
  arêtes, pas le numéro de pas.
* **Commit : +3 411 Mo** sur 400 pas, soit ~8,5 Mo/pas en moyenne. C'est le seul compteur qui
  monte franchement — et c'est lui qui tue.

La forme de la courbe impose le verdict : **RAM stable après la montée initiale, donc pas de
fuite**. Le diagnostic demandé s'arrête là pour la RAM hôte — et se poursuit côté carte.

---

## 3. Où passe la mémoire, alors ? (mesuré)

Banc `scripts/bench_step.py`, batch 16, toy séquence 10, un processus neuf par mesure mémoire.

| Arêtes | Forward | Forward + backward | Pas complet | VRAM réservée | Commit |
|---|---|---|---|---|---|
| 400 | 800 ms | 1 755 ms | 1 828 ms | 380 Mo | 3 312 Mo |
| 2 200 | 1 132 ms | 2 309 ms | 2 547–2 600 ms | 2 892–2 906 Mo | 5 860–5 906 Mo |
| 2 200, `tbptt 4` | — | — | 2 186 ms | **1 240 Mo** | 4 148 Mo |

Les structures que la consigne soupçonnait sont **négligeables** (mesuré, taille théorique) :

| Structure | Taille |
|---|---|
| `edge_index` [2, 20 000] int64 | 320 Ko |
| `edge_usage` [20 000] float32 | 80 Ko |
| `_edge_mask` [20 000] float32 | 80 Ko |
| `_edge_keys` ~11 000 entiers Python | ~0,4 Mo |
| `cemetery` (15 entrées × 12 lignes) | ~30 Ko |
| `history` (400 entrées) | ~50 Ko |

Tout le poids est dans les **activations conservées pour le backward** : 2 892 Mo de VRAM pour
un modèle de ~5 Mo de paramètres. Ordre de grandeur cohérent avec le code : un chunk vaut
`512 × 16 × 32` flottants = 1 Mo, et un pas toy conserve 8 tours × 10 positions × ~4,3 chunks ×
plusieurs tenseurs intermédiaires (`keys`, `queries`, `score`, `message`, `out`).

**Conséquence directe et chiffrée :** avec 8 Go de carte, la VRAM réservée plafonne le câblage
autour de **5 000 à 6 000 arêtes** en batch 16 (2,9 Go à 2 200, croissance à peu près linéaire),
et autour de **2 500 à 3 000 arêtes** en batch 32. Au-delà, Windows (WDDM) sauve la carte en
débordant sur la RAM système : le working set explose, le commit aussi, et le processus est tué
sans traceback. C'est exactement le scénario du run mort : 12 591 Mo de commit, 3 571 Mo de WS,
aucune exception dans le log. **Le GPU n'était donc pas « hors de cause à 0,03 GiB »** : ce
chiffre décrit une mesure où le graphe n'était pas retenu (probablement `no_grad`).

---

## 4. Pourquoi 0,283 s/pas n'est pas reproductible (mesuré)

| Mesure | Valeur mesurée | Écart avec 283 ms |
|---|---|---|
| Forward seul, 2 200 arêtes, batch 16, seq 10 | 1 132 ms | ×4,0 |
| Pas complet, 2 200 arêtes, même config | 2 600 ms | ×9,2 |
| Forward seul, 400 arêtes (graphe 5,5× plus petit) | 800 ms | ×2,8 |
| Pas complet, texte, 1 200 arêtes, seq 128, TBPTT 16 | **22 105 ms** | ×78 |

Aucune configuration mesurable aujourd'hui ne donne 283 ms pour un pas. Le chiffre de la v4 doit
être **retiré**, pas nuancé : il ne dit pas quel étage il chronométrait (forward ? avec quelles
naissances ? quelle séquence ? quel batch ?), et aucun fichier du dépôt ne permet de le refaire.
`docs/RAPPORT-v4.md` §6.3 et `docs/README.fr.md` sont corrigés en conséquence.

Décomposition honnête du pas à 2 200 arêtes : **1 132 ms de forward** (43 %), **1 177 ms de
backward**, le reste (naissances, taxes, clip, optimiseur, cycle de vie) est marginal. Le
forward coûte ~14 ms par tour de diffusion, soit ~3 ms par chunk-tour : c'est du coût de
lancement de kernels et de synchronisations (`torch.unique`, `bool(keep.any())`, `nonzero`,
boucle Python sur les chunks) sur des tenseurs minuscules, pas du calcul utile. Le profil v3
disait déjà 52 % dans `run_backward` et 16 % dans le forward sur un autre découpage ; le
chiffre qui compte ici est qu'**aucun des deux n'est optimisé**.

---

## 5. Réponse aux quatre questions de la consigne

1. **RAM ∝ arêtes, ou fuite, ou stable ?** → **Stable** : +26 Mo pour +2 334 arêtes. Pas de
   fuite. La croissance est en VRAM (activations), pas en RAM hôte.
2. **`gc.collect()` réduit-il la RAM ?** → **Non**, 0 Mo à chaque rapport : aucun cycle à
   récupérer. Un `gc.collect()` périodique est donc inutile ici (il est mesuré, pas deviné).
3. **Le batch 32 était-il la cause ?** → Il en était **l'aggravant décisif** : la VRAM double
   avec le batch, donc le mur tombe deux fois plus tôt (2 500–3 000 arêtes au lieu de
   5 000–6 000). Mais baisser le batch ne supprime pas le mur : il le repousse, comme annoncé.
4. **Pourquoi le débit réel est-il 4 à 8× plus lent que la mesure d'acceptation ?** → Parce que
   cette mesure ne chronométrait pas un pas. Le débit réel est 2,6 s/pas à 2 200 arêtes (toy) et
   22,1 s/pas à 1 200 arêtes (texte).

---

## 6. ETA du run texte 10 000 pas — et pourquoi elle est disqualifiante

Base mesurée : **22,1 s/pas** à 1 200 arêtes, séquence 128, batch 16, TBPTT 16. Croissance du
câblage : ~5,7 arêtes/pas (mesurée sur le toy, **(supposé)** comparable sur le texte).

| Hypothèse | Coût |
|---|---|
| Débit constant à 22,1 s/pas (plancher irréaliste) | **61 h** |
| Coût moyen ∝ arêtes (28 500 arêtes en moyenne) | ~525 s/pas en moyenne, soit **~61 jours** |
| Coût ∝ arêtes, 57 000 arêtes au pas 10 000 **(estimé)** | > 1 000 s/pas en fin de run, total de l'ordre de **plusieurs semaines** (extrapolation grossière, pas une mesure) |

À cela s'ajoute le mur de VRAM : à 1 200 arêtes le texte tient (1 592 Mo réservés), mais le
débit par arête y est déjà 6× celui du toy, et le run texte dépasse 6 000 arêtes dès le pas
~1 000. **Le run texte 10 000 pas n'est pas lançable en l'état** — ni en 6 h, ni en une nuit.

Trois leviers, chacun chiffrable avant engagement :

1. **`tbptt 4` au lieu de 16** : mesuré sur le toy, VRAM ÷ 2,3 et pas − 16 %. Gain mémoire réel,
   changement de crédit temporel assumé (le run reste valide, il apprend moins loin).
2. **Checkpointing de gradient par tour de diffusion** : mémoire ÷ ~5 attendu, +30 % de temps
   **(estimé, non mesuré)** — c'est le levier qui garderait la sémantique intacte.
3. **Durée réduite** : 5 000 pas à 22 s = **31 h** au plancher. Toujours pas une nuit.

Décision humaine requise avant tout lancement : le choix change soit la sémantique
d'apprentissage (1), soit le code du forward (2), soit la portée du verdict (3).

---

## 7. Ce qui est corrigé dans le dépôt

* `mycelia/train.py` : instrumentation RAM/pic/commit/VRAM + sonde `gc.collect()` à chaque
  rapport, `ram_mb` dans `history.json`.
* `configs/config.yaml` : `toy_batch: 32` → **16**, avec la raison en commentaire.
* `scripts/bench_step.py` : banc reproductible (forward / fwd+bwd / pas complet, `--data`,
  `--tbptt`, `--only step`), un processus par mesure mémoire.
* `docs/README.fr.md` : les deux mesures expliquées côte à côte, le chiffre de 283 ms retiré.
* Aucun test modifié : `pytest -q` → **41 passed** après instrumentation.

---

## 8. Nouveau critère (f), et sa formulation honnête

Le critère demandé — « `ram_mb` stable en régime » — est **tenu** par le run d'observation
(+26 Mo sur 400 pas). Mais il est insuffisant : la RAM était stable pendant que le commit
montait de 3,4 Go, et c'est le commit qui a tué le run précédent. La formulation proposée est
donc :

> **(f) stabilité mémoire** : `ram_mb` stable **et** `commit_mb` sans croissance monotone
> inexpliquée sur la durée du run ; toute croissance du commit doit être attribuée (cache CUDA,
> débordement WDDM) et documentée.

Sans le second terme, le critère aurait été vert sur un run mourant.
