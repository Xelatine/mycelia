# Mycelia v3 — rapport de corrections et de mesures

Date : 3 novembre 2026. Périmètre : `C:\Users\alexw\Downloads\mycelia`, corrections C1 à C8,
tests, mesures sur RTX 3070 (torch 2.11.0+cu128, Python 3.12.10).

Toutes les valeurs de ce rapport sont **mesurées** sur cette machine. Les cases « non mesuré »
le disent explicitement : rien n'est extrapolé sans être signalé.

---

## 1. Résumé exécutif

| Point | État | Preuve |
|---|---|---|
| C1 verrou anti-double-lancement | fait, vérifié en conditions réelles | second lancement refusé, PID affiché |
| C2 sauvegarde horloge + history atomique | fait, vérifié | `checkpoint (premier) à step 10, 3 s de run` |
| C3 slots alloués nés morts | fait, vérifié dans un run | `nœuds 514` au lieu de 532 |
| C4 performance | **partiel : cible < 2 s/pas NON atteinte** | 0,29 s/pas sans naissances ; 8,78 s/pas avec 8 |
| C5 logs sans ambiguïté | fait, vérifié | `arêtes_nées 400 / nœuds_nés 2`, ms glissant |
| C6 GrowthManager transparent | fait | messages explicites par événement |
| C7 honnêteté de la doc | fait | README et doc FR mis à jour |
| C8 décodage propre | fait | `clean_text` à l'affichage seulement |
| Tests | **36 passent** | 34 existants + 2 nouveaux, 46,95 s |
| Run toy 500 pas | terminé | perte 3,54 → 0,25 ; `0 -> 2 -> 1` |
| Run toy 2 000 pas complet | **non fait** | — |
| Run texte | **partiel et arrêté** : ~100 pas pour mesurer 8,5 s/pas, puis stoppé (règle C4) | logs du run v2 |
| Run texte 10 000 pas | **non lancé** | — |
| ZIP | `mycelia-v3.zip` | 37 fichiers, code + tests |

---

## 2. Ce que le run réel a appris (avant corrections)

Le run texte v2 (batch 16, séquence 128, 8 tours, 512 nœuds, vocab 10 000) a été lancé,
mesuré jusqu'au pas 100 puis **arrêté volontairement** pour corriger les défauts de la v3.
Ses chiffres sont donc réels mais partiels — aucun run texte long n'a été mené à terme.
Il a produit :

| Pas | Perte | Perplexité | Arêtes | Nœuds | ms/pas affiché |
|---|---|---|---|---|---|
| 20 | 9,3269 | 11 236 | 174 | 532 | 3 166 |
| 40 | 9,2568 | 10 475 | 334 | 532 | 3 511 |
| 60 | 9,1529 | 9 441 | 494 | 532 | **29 231** (deux processus concurrents) |
| 100 | 8,9545 | 7 742 | 814 | 532 | 6 899 |

`ms/pas` était calculé `temps écoulé / pas` : une moyenne cumulée, qui ne peut que monter et
qui masque tout intervalle lent. Le chiffre réel de l'intervalle 51–100 : 424,6 s pour 50 pas,
soit **8,5 s/pas** — et non les 3,3 s/pas des 40 premiers pas, quand le graphe était presque vide.

**Deux entraînements ont tourné en parallèle** (PIDs 13440 à 22:15:37 et 9972 à 22:44:22, même
ligne de commande), tous deux écrivant `data/genome.pt` et `data/genome_latest.pt`, d'où le
pic à 29 s/pas. Aucun verrou n'empêchait cela : c'est l'origine de C1.

Ce que le run a aussi invalidé : l'idée que « naissances » comptait des nœuds. `naissances`
valait `ne - 12`, c'est-à-dire des **arêtes**, alors que la ligne voisine parlait de nœuds.
D'où C5.

---

## 3. Les huit corrections

### C1 — Verrou anti-double-lancement (`mycelia/train.py`)

`acquire_lock()` écrit `data/train.lock` avec le PID courant ; si le fichier existe et que son
PID est **vivant**, le programme refuse de démarrer :

```
Un entraînement tourne déjà (PID 4100). Supprimez C:\Users\alexw\Downloads\mycelia\data\train.lock si c'est une erreur.
```

`release_lock()` est appelé dans un `finally`, et ne supprime le fichier que s'il contient
**notre** PID. Un verrou périmé (PID mort) est signalé puis remplacé.

Détail Windows important : `os.kill(pid, 0)` n'est **pas** une sonde sous Windows, le second
argument y est un code de sortie et l'appel tue le processus visé. `pid_alive()` utilise donc
`OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` + `GetExitCodeProcess` et compare à
`STILL_ACTIVE` (259). Sous Unix, `os.kill(pid, 0)` est utilisé.

Vérifié deux fois : refus pendant un run, et disparition du fichier après la fin normale du
run toy 500 (aucun `data/train.lock` restant, aucun processus).

### C2 — Sauvegarde à l'horloge, historique incrémental

- `training.save_seconds: 600` remplace `save_every: 1000` comme règle ; `training.first_save_step: 10`
  force un checkpoint très tôt.
- `checkpoint(reason)` écrit `genome.pt` + `genome_latest.pt`, puis l'historique, et journalise
  `checkpoint (premier|horloge|interruption|fin de run) à step N, T s de run`.
- `write_history()` écrit `history.json.tmp` puis `os.replace` : le cockpit ne lit jamais un
  fichier à moitié écrit, et un arrêt brutal ne perd rien de déjà rapporté.
- `--save-every N` reste disponible comme déclencheur supplémentaire, pas comme mécanisme
  principal.

Vérifié : `checkpoint (premier) à step 10, 3 s de run`.

### C3 — Les slots alloués naissent morts (`mycelia/brain.py`)

Avant : `grow_population` faisait `alive = torch.ones(new_n)`. Un message adressé au nœud 531
allouait 20 slots et en rendait **19 vivants d'office**, avec date de naissance et gel : une
violation directe de F3 (« le message précède l'existence »), et un `node_alive` gonflé qui
fausse l'échantillonnage de redondance et les signes vitaux.

Après : les nouveaux slots sont une **réserve morte** ; seul `growth.try_birth_node` réveille
le slot réellement adressé (bruit faible, gel de 200 pas), et incrémente `node_births`.

Vérifié dans le run toy réel :

```
step 50 | ... | nœuds 514 | arêtes_nées 400 | nœuds_nés 2 | ...
```

512 nœuds + 2 vraies naissances = 514, alors que l'ancien code affichait 532 pour 2 naissances.
Test : `test_allocated_slots_dead` (C3.1).

### C4 — Performance : ce que la mesure dit vraiment

**La prémisse de la prescription est fausse pour ce code.** L'état n'est pas un dict Python
`node_id → tenseur` : la membrane est déjà **un seul tenseur `[N, batch, dim]`**
(`_fresh_state`), mise à jour par `index_add` / `index_put` sur le tenseur entier.

**Le profil** d'un pas (2 000 arêtes, sans naissance, batch 8, séquence 32, 1,59 s) :

| Poste | Temps | Part |
|---|---|---|
| `run_backward` | 0,830 s | 52 % |
| `forward` (Python) | 0,249 s | 16 % |
| `NodeBank.forward_rows` | 0,072 s | 4,5 % |
| `baddbmm` (972 appels) | 0,061 s | 3,8 % |
| `torch._unique2` (199 appels) | 0,037 s | 2,3 % |
| `nonzero` | 0,019 s | 1,2 % |
| `torch.isin` | 0,014 s | 0,9 % |

Un pas texte = 128 positions × 8 tours = **1 024 itérations séquentielles**, quelques dizaines
de kernels chacune. Ce n'est ni un dict, ni une matrice dense : c'est le nombre de kernels.

**Mesures en config cible** (batch 16, séquence 128, 8 tours, 500 arêtes) :

| Naissances autorisées | Temps/pas | Pic mémoire |
|---|---|---|
| 0 | **0,290 s** | 0,03 GiB |
| 1 | **0,238 s** | 0,03 GiB |
| 8 | **8,781 s** | 0,60 GiB |

Le saut est entre 1 et 8 : ×37 pour sept naissances de plus, alors que 0 → 1 est gratuit. Le
coût du run texte (8,5 s/pas) vient donc du **chemin des naissances**
(`_birth_edges`, appelé à chaque tour avec échantillonnage, top-k et sélection côté hôte), pas
du passage de messages.

**Mémoire** (le mur à 2 000 arêtes) :

| Arêtes | Config | Temps/pas | Pic |
|---|---|---|---|
| 500 | batch 8, seq 32, naissances 0 | 0,080 s | 0,03 GiB |
| 1 000 | batch 8, seq 32, naissances 0 | 2,049 s | 0,16 GiB |
| 2 000 | batch 8, seq 32, naissances 0 | 2,235 s | 4,04 GiB |
| 2 000 | batch 16, seq 128 | **OOM** | 14,43 GiB demandés sur 8 GiB |

**Ce qui a été fait** : suppression du `torch.isin` trié du chemin chaud (remplacé par un
ensemble côté hôte `_edge_keys`, à pas stable `max_population`, reconstruit après élagage et
au chargement d'un génome) ; compteur de naissances réelles ; affichage du temps par moyenne
glissante. **Ce qui n'a pas été fait, volontairement** : regrouper le calcul de tous les nœuds à
chaque tour — cela multiplierait la mémoire d'activations qui provoque déjà l'OOM.

**Verdict** : la cible « < 2 s/pas à 2 000 arêtes » **n'est pas atteinte**. Elle est plausible
en traitant le budget des naissances (le seul poste qui explique 8,78 s), pas en changeant la
représentation de l'état. `test_scaling_edges` (C4.4, ratio < 3× entre 500 et 2 000 arêtes)
passe sur câblage structuré ; il n'est pas validé sur le cas dense qui OOM.

### C5 — Logs sans ambiguïté (`mycelia/train.py`)

- Deux compteurs distincts : `arêtes_nées` (`ne - 12`, arêtes) et `nœuds_nés` (vraies
  naissances issues de `try_birth_node`). Les deux sont aussi dans `history.json`.
- `ms/step` = moyenne des 20 derniers pas, avec le nombre de pas mesurés :
  `479 ms/step (moyenne des 20 derniers)` … puis `2199` quand le câblage grossit.

### C6 — GrowthManager transparent (`mycelia/growth.py`)

Le premier `update` (référence absente) journalise :
`référence de validation posée à step 500 (val=…) — première décision au pas 1000`.
Les événements disent ce qu'ils font, et rien de plus :
`phase CROISSANCE … : naissances ouvertes`, `phase OPTIMISATION … : naissances suspendues`,
`PALIER : pool 512 → 768 …`. Aucun événement ne laisse croire à une action qui n'a pas lieu :
seul `reallocate` touche le pool.

### C7 — Honnêteté de la doc (`README.md`, `docs/README.fr.md`, `configs/config.yaml`)

« ~5–8 s/pas » est remplacé par les mesures réelles et leur évolution (3,2 s/pas au début,
8,5 s/pas vers 800 arêtes), la cible non atteinte, la cause (nombre de kernels), et
**l'ETA inconnue** tant que ce n'est pas réglé.

### C8 — Décodage propre, à l'affichage (`mycelia/data.py`, `scripts/generate.py`, `dashboard/runner.py`)

`clean_text()` retire U+FFFD et les caractères de catégorie Unicode « C » (contrôle), en
conservant espaces, tabulations et retours à la ligne. Utilisé dans `generate.py` (texte complet
et dernier token) et dans le chat du cockpit. Rien n'est modifié dans le modèle : le cerveau
n'est pas maquillé.

---

## 4. Tests

```
python -m pytest tests -q
36 passed, 1 warning in 46.95s
```

- 34 tests existants conservés, dont `test_resume_continuity` (tolérance **1e-6 inchangée**).
- 2 tests nouveaux : `test_allocated_slots_dead` (C3.1) et `test_scaling_edges` (C4.4).
- `test_newborn_gradients_frozen` a été adapté à la nouvelle sémantique (le slot doit être
  réveillé par `try_birth_node` avant d'être utilisé), **sans** relâcher la moindre assertion.
- `test_roundtrip` vérifie `format_version == 3`.

**Bug trouvé et corrigé pendant la v3** : l'ensemble `_edge_keys`, dérivé de l'état et non
sauvegardé, n'était pas reconstruit au chargement du génome. La reprise rouvrait alors des
arêtes existantes et `test_resume_continuity` dérivait de 5,2e-3. Corrigé dans
`brain_from_genome` par `_rebuild_edge_keys()` ; le test repasse à 1e-6 sans être touché.

**Checkpoint format 3** : `counters.node_births` ajouté ; le lecteur accepte encore les
génomes v2 (champ absent → 0), donc les anciens fichiers restent lisibles.

---

## 5. Validation demandée : où elle en est

1. `pytest` complet — **fait**, 36/36. ✔
2. Run toy 500 pas — **fait** : perte 3,5449 (pas 50) → 0,2530 (pas 400), arêtes 412 → 2 826,
   arêtes coupées 161 puis 225 puis 283, nœuds enterrés 124 puis 11, cimetière jusqu'à 50,
   `nœuds 514` et `arêtes_nées`/`nœuds_nés` distincts, ms glissant 479 → 2 199.
   Fin normale : `chemin : 0 -> 2 -> 1`, génome écrit, verrou relâché. ✔
3. Mesure de vitesse à ~2 000 arêtes, < 2 s/pas exigé — **fait mais ÉCHEC** : OOM en config
   cible, 2,24 s/pas à batch 8 / séquence 32. ✘
4. Run toy 2 000 pas complet — **non lancé** (la règle 3 n'est pas satisfaite). ✘
5. Run texte 10 000 pas — **non lancé**. Le seul run texte existant est partiel (~100 pas,
   arrêté après la mesure de vitesse), et n'est pas présenté comme un résultat. ✘

---

## 6. Ce qu'il reste à faire, par ordre de rentabilité

1. **Budget des naissances** : le seul poste qui explique 8,78 s/pas. Pistes mesurables :
   n'appeler `_birth_edges` qu'aux tours où le budget le justifie, vectoriser la sélection
   (aujourd'hui une boucle Python sur les paires), ou amortir le budget sur le pas
   d'optimisation plutôt que sur chaque tour — à décider explicitement, car cela change *qui*
   peut naître.
2. **Mur mémoire à 2 000 arêtes** : accumuler les messages par blocs d'arêtes bornés
   (chunking) plutôt que de matérialiser `[E, B, dim]` pour toutes les arêtes chaudes.
3. Puis seulement : run toy 2 000 pas avec les cinq critères de verdict, et le run texte.
4. Cosmétique restant : le libellé « naissances » du panneau du cockpit
   (`dashboard/static/index.html`) n'a pas été renommé.

---

## 7. Commandes utiles

```bash
python -m pytest tests -q                                  # 36 tests
python mycelia/train.py --steps 500 --report-every 50      # toy court
python mycelia/train.py --data text --scale 0              # texte (après C4)
python .cache/bench2.py                                    # mesure de vitesse C4
```

Le ZIP `mycelia-v3.zip` contient `mycelia/`, `tests/`, `scripts/`, `dashboard/`, `configs/`,
`docs/`, plus `README.md`, `requirements.txt`, `LICENSE` et `.gitignore` — sans `data/`,
`archive/` ni `__pycache__`.
