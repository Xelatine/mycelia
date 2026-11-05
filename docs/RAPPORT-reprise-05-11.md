# Mycelia — rapport du 5 novembre 2026

Date de rédaction : 5 novembre 2026, 20 h 25 (heure locale).
Périmètre : `C:\Users\alexw\Documents\mycelia`, reprise après l'incident de suppression.
Machine : RTX 3070 Laptop, 31,2 Go de RAM (limite de commit 32,2–35,2 Go), torch 2.11.0+cu128,
Python 3.12.10, disque 21 Go libres.

Ce rapport ne rapporte que ce qui a été **mesuré ou constaté dans ce dossier**. Trois étiquettes
sont utilisées quand la preuve manque : **(mesuré)**, **(estimé)**, **(supposé)**.
Il remplace le `RAPPORT-COMPLET.md` annoncé dans la consigne de reprise : ce fichier **n'existe
pas** ici (recherche par nom et par contenu dans tout le dossier hors `.git`, aucune occurrence).

---

## 1. Verdict en une phrase

**La validation s'arrête au jouet : le run toy n'a pas fini — il est mort au pas 1502 sur 2 000,
sans traceback ni sortie propre — et ses critères (d) vitesse et (e) fin propre échouent.** Conformément
à la règle de la consigne (« si un critère échoue : STOP, diagnostic exact avec preuve »), le run
texte 10 000 pas et la baseline Transformer n'ont **pas** été lancés, et le Palier 1 n'est **pas**
évalué. Ce qui est acquis : 41 tests verts, un dépôt git propre, et un diagnostic causal du
ralentissement (`toy_batch: 32` → saturation mémoire → OOM → batch 16).

| Critère du toy | Verdict | Mesure |
|---|---|---|
| (a) perte qui baisse nettement | **✔** | 2,9086 → 0,3724 entre les pas 100 et 300 (facteur 7,8) |
| (b) `arêtes_nées` ≠ `nœuds_nés` | **✔** | 11 134 arêtes nées vs 6 vrais nœuds nés au pas 1500 |
| (c) nœuds vivants < pool | **✔** | 518 vivants pour un pool de 587 au pas 1500 |
| (d) ms/pas en régime < 2 000 | **✗** | 2 018 → 3 774 ms/pas après le pas 762 ; jamais durablement sous 2 000 |
| (e) fin propre `0 -> 2 -> 1`, verrou relâché, `genome.pt` écrit | **✗** | mort au pas 1502, `data/train.lock` toujours présent avec le PID 15108 (mort) ; `genome.pt` écrit mais au pas 1502 |

---

## 2. Chronologie mesurée du run

Commande : `python mycelia/train.py --steps 2000 --report-every 100` (processus PID 15108).
Log : `logs_toy_v4_2000_rerun.txt`, 50 lignes, dernier caractère écrit à 02:02.
Configuration effective : `toy_batch: 32` (`configs/config.yaml:40`), réduite à 16 en cours de run.

| Pas | Perte | ppl | Arêtes | Nœuds vivants | arêtes_nées | nœuds_nés | ms/pas (moy. 20) |
|---|---|---|---|---|---|---|---|
| 100 | 2,9086 | 18,3 | 792 | 514 | 780 | 2 | 921 |
| 200 | 1,4417 | 4,2 | 1 541 | 514 | 1 547 | 2 | 1 017 |
| 300 | 0,3724 | 1,5 | 2 207 | 514 | 2 314 | 2 | 1 248 |
| 400 | 0,0771 | 1,1 | 2 760 | 503 | 3 088 | 2 | 1 471 |
| 500 | 0,0109 | 1,0 | 3 254 | 511 | 3 858 | 2 | 3 288 |
| 600 | 0,0057 | 1,0 | 3 678 | 515 | 4 607 | 4 | 3 936 |
| 700 | 0,0023 | 1,0 | 4 058 | 515 | 5 355 | 4 | 4 611 |
| 800 | 0,0019 | 1,0 | 4 410 | 515 | 6 113 | 4 | 2 018 |
| 900 | 0,0034 | 1,0 | 4 712 | 514 | 6 856 | 4 | 2 071 |
| 1 000 | 0,0007 | 1,0 | 4 986 | 516 | 7 601 | 4 | 2 157 |
| 1 100 | 0,0007 | 1,0 | 5 174 | 518 | 8 288 | 6 | 2 181 |
| 1 200 | 0,0005 | 1,0 | 5 369 | 518 | 9 000 | 6 | 2 496 |
| 1 300 | 0,0063 | 1,0 | 5 549 | 518 | 9 717 | 6 | 2 266 |
| 1 400 | 0,0008 | 1,0 | 5 715 | 518 | 10 438 | 6 | 2 336 |
| 1 500 | 0,0004 | 1,0 | 5 839 | 518 | 11 134 | 6 | 3 774 |

Événements du log, dans l'ordre :

* verrou périmé PID 8848 détecté et remplacé par PID 15108 (C1 exercé en réel) ;
* pas 500 : référence de validation posée (`val = 1,0`), première décision de croissance reportée
  au pas 1 000 ;
* pas 500 et 1 000 : `optimiseur reconstruit après croissance (6 tenseurs re-attachés)` ;
* pas 500 : `pool 512 → 532` pour **2 vraies naissances** ; pas 1 000 : `pool 555 → 587` pour
  2 de plus (compteurs distincts, C3/C5) ;
* pas 762 : **`OOM: batch réduit à 16 (le graphe reste creux)`** ;
* pas 1 500 : **`phase OPTIMISATION`** (`val = 1,0`), naissances suspendues — `nœuds_nés` reste
  figé à 6 ensuite ;
* dernier événement : `checkpoint (horloge) à step 1502, 7240 s de run`, puis plus rien.

**Nuance de lecture :** la ligne de signes vitaux est construite **avant** le cycle de vie du même
pas. `morts-arêtes`, `morts-nœuds` et `cimetière` ont donc un pas de retard. Au pas 300, la ligne
affiche `morts-arêtes 119` et `cimetière 0` alors que le cycle vient de couper 221 arêtes et
d'enterrer 51 nœuds. Ce n'est pas une incohérence, c'est l'ordre d'exécution.

**Convergence :** perte 0,0004 et ppl 1,0 au pas 1 500 signifient que le jouet est **convergé**,
pas que le modèle a appris quoi que ce soit d'utile. Le corpus toy est un jouet de câblage ;
aucune conclusion scientifique ne se tire de ppl ≈ 1,0.

---

## 3. Pourquoi le run ralentit, puis pourquoi il est mort

### 3.1 Falaise de vitesse — cause identifiée (mesuré)

`toy_batch: 32` porte la mémoire validée du processus à **12 591 Mo** (WS 3 571 Mo) pour une limite
de commit de 35,2 Go, dont il ne restait que **0,0–0,2 Go**. Conséquences mesurées sur la machine :
`import torch` échoue en `OSError: [WinError 1455] Le fichier de pagination est insuffisant…
cublas64_12.dll`, `fork` de Git Bash échoue (`Resource temporarily unavailable`), PowerShell ne
démarre plus son CLR (`HRESULT 80004005`), `wmic`/`Get-CimInstance` inutilisables.

Sous cette pression, les ms/pas montent de 1 471 (pas 400) à 4 611 (pas 700, soit ×3,1). Au pas
762, l'OOM fait tomber le batch à 16 : le processus retombe à **812 Mo de WS** et le débit revient
à 2 018 ms/pas. Le run v4 précédent (`logs_toy_v4.txt`) montre la même marche (1 799 → 4 678 ms/pas
vers 3 247 arêtes) : le mécanisme est **reproductible**.

**Réserve de comparabilité, à ne pas oublier :** les pas 100–700 tournent en batch 32, les pas 800+
en batch 16. Les pertes des deux segments ne se comparent pas directement (la variance par pas
augmente visiblement en batch 16 : 0,0063 au pas 1 300 entre 0,0005 et 0,0008).

### 3.2 Mort du run au pas 1502 (constaté ; cause supposée)

Constats : le PID 15108 n'existe plus ; le log s'arrête net après le checkpoint d'horloge du pas
1502, **sans traceback, sans ligne de fin, sans message d'interruption** ; `data/train.lock`
contient encore `15108` ; `data/genome.pt` et `data/genome_latest.pt` datent de 02:02 (45 934 037
octets), soit exactement l'heure du dernier checkpoint.

Le run n'a donc pas été interrompu par une exception Python (rien dans le log) : il a été **tué de
l'extérieur**. Cause la plus probable **(supposé, non prouvé)** : la même saturation de mémoire
que §3.1 — la vitesse remonte à 3 774 ms/pas au pas 1 500, signe d'une seconde montée de pression,
et Windows termine un processus sans traceback quand le commit est épuisé. Un arrêt manuel ou un
kill par l'utilisateur n'est pas exclu ; le log ne permet pas de trancher.

**Ce que cela coûte :** 1 502 pas sur 2 000 (75 %), 7 240 s (2 h 01) de calcul, et le critère (e)
manqué pour la seule raison qui compte — il n'y a pas eu de sortie propre.

---

## 4. Ce que le toy démontre quand même

Ces points sont mesurés dans le log et restent valables indépendamment de la mort du run :

* **Compteurs distincts (C5)** : 11 134 arêtes nées pour 6 vrais nœuds nés au pas 1 500. Un compteur
  unique aurait annoncé 11 134 « nœuds » ; le code ne le fait pas.
* **Slots morts (C3)** : 518 nœuds vivants pour un pool de 587 au pas 1 500 — l'écart est réel et
  persistant, pas un artefact d'affichage.
* **Cycle de vie complet, dans les deux sens** : 51 nœuds enterrés au pas 300 (grâce d'isolement
  écoulée) avec un cimetière à 11 au pas 400, donc **40 nœuds rappelés par un message** (F3). Ce
  n'est pas une perte, c'est la résurrection en action.
* **Élagage continu** : 368 → 584 arêtes coupées par tranche de 100 pas entre les pas 600 et 1 400.
* **Fin des enterrements** : `morts-nœuds` reste figé à 68 des pas 900 à 1 500, cimetière à 0 : une
  fois le graphe saturé et toutes les naissances suspendues, plus aucun nœud n'est isolé.
* **Croissance bornée par la phase** : deux poussées de 20 puis 32 slots, puis `phase OPTIMISATION`
  au pas 1 500 qui gèle les naissances. La boucle croissance/optimisation fonctionne.

---

## 5. Ce qui n'a pas été lancé, et pourquoi

| Élément | État | Raison |
|---|---|---|
| Run texte 10 000 pas | **non lancé** | critère (d) échoué ; de plus l'ETA réelle est d'un autre ordre que prévu (§5.1) |
| Run texte — échantillons 40 tokens aux pas 200/1 000/3 000/5 000/10 000 | **non collectés** | dépend du run texte |
| `docs/echantillons-texte.md` | **non écrit** | idem |
| Baseline Transformer 10 000 pas | **non lancée** | STOP ; et `baseline_ppl.txt` reste absent |
| `docs/BILAN-run-texte.md` | **non écrit** | aucun chiffre de run texte ni de baseline à comparer ; l'écrire aurait été de la fiction |
| Verdict Palier 1 | **non évalué** | dépend du run texte et de la baseline |
| Table 3 seeds | **non remplie** | dépend des deux runs |

### 5.1 ETA : l'hypothèse de 47 min ne tient pas (estimé)

Le plan tablait sur 0,283 s/pas. Sur cette machine et cette configuration, le run toy mesure
**1,25 s/pas à 2 207 arêtes** et **2,34 s/pas à 5 715 arêtes** : le facteur d'écart avec 0,283 s/pas
est de 4 à 8. Les arêtes croissent d'environ 5,7 par pas ; à 10 000 pas le graphe atteindrait
~56 000 arêtes. En supposant un coût par pas à peu près proportionnel au nombre d'arêtes (hypothèse
grossière, non mesurée au-delà de 5 839 arêtes), le run texte coûterait **de l'ordre de plusieurs
dizaines d'heures**, pas une heure. Cette estimation est un ordre de grandeur, pas une prévision :
elle doit être remesurée avant tout lancement.

---

## 6. Git et livraison

| Commit | Contenu |
|---|---|
| `db1ab80` | commit initial (129 fichiers suivis) |
| `a1413ee` | tests |
| `558c038` | `.gitignore` + `git rm --cached data/train.lock .dash.pid` — « gitignore: runtime locks (verrous d'exécution, pas de contenu) » |
| `d0d0957` | « archive: ZIP horodaté post-reprise » (`mycelia-v4-2026-11-04.zip`) |
| `47458c8` | « docs: rapport de reprise du 5 novembre… » (première version du présent fichier) |

`.gitignore` couvre désormais `data/train.lock`, `data/*.pt`, `baseline_ppl.txt`, `history.json`,
`.dash.pid`, `logs_*.txt`. Aucun autre nettoyage n'a été fait : la consigne l'interdit sans accord
explicite. La version finale de ce rapport et le ZIP du 5 novembre sont livrés dans des commits
séparés (voir `git log`).

---

## 7. Contrôles finaux

| Contrôle | Résultat | Preuve |
|---|---|---|
| `python -m pytest tests -q` | **41 passed, 1 warning, 73,46 s, exit 0** | exécuté après la fin du run, mémoire libérée (RAM 17,5/31,2 Go, commit 15,7/32,2 Go) |
| Continuité de reprise à 1e-6 | **verte** | `tests/test_checkpoint.py:97`, `test_resume_continuity` |
| Arbre git | **propre** après commit du rapport et du ZIP | `git status --porcelain` |
| Run toy documenté | **oui**, avec sa mort et ses chiffres | `logs_toy_v4_2000_rerun.txt`, `history.json` (16 entrées jusqu'au pas 1 500) |
| Bilan écrit | **oui** (ce rapport, avec verdict négatif assumé) | ce fichier |
| Verrou relâché | **non** | `data/train.lock` = `15108`, processus mort |

Le warning pytest est une dépréciation Starlette/httpx (`fastapi.testclient`), sans effet sur les
tests. Le code n'a **pas** été modifié pendant cette session : seuls `.gitignore`, l'index git et
`docs/` ont changé, donc les 41 tests valident exactement ce qui est livré.

---

## 8. État sur disque et reprise possible

* `data/genome.pt` / `data/genome_latest.pt` : 45 934 037 octets, écrits le 5 novembre à 02:02,
  soit le checkpoint d'horloge du **pas 1502** — pas le génome final des 2 000 pas.
* `history.json` : entrées jusqu'au pas 1 500, dont l'événement `optimize` et les métriques du pas.
  Pas d'entrée de fin.
* `data/train.lock` : contient `15108`, processus inexistant. **Verrou périmé.** Il n'a pas été
  supprimé (règle : ne rien nettoyer sans accord).
* Reprise : `python mycelia/train.py --resume` est le chemin prévu ; `acquire_lock` vérifie
  `pid_alive` avant de refuser (`mycelia/train.py:127`, `pid_alive` via `OpenProcess`, jamais
  `os.kill(pid, 0)`), donc un verrou périmé est remplacé automatiquement. Il reste **498 pas** à
  parcourir (~19 à 32 min au débit mesuré, estimé), **mais** la reprise recharge `toy_batch: 32` :
  sans changement de configuration, le run repartira dans la même saturation et mourra de la même
  façon. C'est le point à trancher avant de relancer.

---

## 9. Prochaine action recommandée

1. **Baisser `toy_batch` de 32 à 16 dans `configs/config.yaml`** (une valeur de configuration, pas
   du code) — c'est la seule variable identifiée qui explique la saturation, et le run a déjà
   tourné 700 pas à 16 sans incident.
2. **Reprendre le toy** avec `--resume` (le verrou périmé sera remplacé) ou, si l'on veut une courbe
   homogène en batch 16 de bout en bout, le relancer sur 2 000 pas (~40 à 75 min estimés au débit
   mesuré). Le second choix coûte 2 h de calcul mais donne un critère (d) interprétable, ce que le
   run actuel ne permet pas.
3. **Ne lancer le texte qu'après** un toy fini proprement : l'ETA doit être remesurée sur les
   premiers 500 pas réels, pas déduite du chiffre de 0,283 s/pas.
4. Faire tourner la baseline Transformer **en parallèle du texte**, jamais pendant le toy, pour ne
   pas rejouer la saturation mémoire.

---

## 9 bis. Addendum du 5 novembre, 21 h — le diagnostic qui corrige §3

Le mécanisme de §3 était partiellement faux : la mémoire qui tue n'est **pas** la RAM hôte.
Mesures de la session de diagnostic (`docs/DIAGNOSTIC-memoire-05-11.md`) :

* **RAM résidente plate** : +26 Mo pour +2 334 arêtes sur 400 pas (1 690 → 1 716 Mo) ;
  `gc.collect()` rend 0 Mo. Aucune fuite, aucune rétention d'objets Python.
* **VRAM proportionnelle au câblage** : 380 Mo réservés à 400 arêtes, **2 892 Mo à 2 200**
  (pas complet, batch 16, 10 positions) pour un modèle de ~5 Mo. Les structures (`edge_index`,
  `_edge_mask`, `_edge_keys`, cimetière) pèsent moins de 1 Mo : tout est dans les activations
  retenues pour le backward.
* **Le commit qui monte** (~8,5 Mo/pas en moyenne, base CUDA incluse) est la conséquence du
  cache CUDA et, au-delà de 8 Go, du débordement WDDM vers la RAM système — d'où la mort sans
  traceback à 12 591 Mo.
* **`tbptt 4` au lieu de 16** (toy) : VRAM 2 892 → **1 240 Mo**, pas 2 600 → **2 186 ms**.
* **Débit réel** : 2,6 s/pas à 2 200 arêtes (toy), **22,1 s/pas** à 1 200 arêtes (texte, seq 128).
  Le 0,283 s/pas de la v4 n'est pas reproductible et est retiré de la documentation.
* ETA texte recalculée : **61 h au plancher**, de l'ordre de **plusieurs semaines** en
  extrapolant la croissance des arêtes (estimation grossière). Le run texte n'est pas lançable en
  l'état ; trois leviers chiffrés sont proposés dans le diagnostic (§6).
* **Critère (f) proposé** : `ram_mb` stable **et** `commit_mb` sans croissance monotone
  inexpliquée — la RAM seule aurait validé un run en train de mourir.

Relance du toy : `configs/config.yaml` passe à `toy_batch: 16`, run de zéro sur 2 000 pas lancé
le 5 novembre à 21 h (`logs_toy_v5_2000_batch16.txt`) avec `ram_mb`/`commit_mb`/VRAM dans chaque
ligne de signes vitaux. Au pas 700 il tenait : RAM 1 706 Mo (plate depuis le pas 100), VRAM
4 062 Mo, commit 7 170 Mo, 2 125 ms/pas, perte 0,0065. Courbe complète et verdict provisoire des
six critères : `docs/DIAGNOSTIC-memoire-05-11.md` §8.

## 10. Limites de ce rapport

* La cause de la mort du run au pas 1502 est **supposée** (saturation mémoire) : le log ne contient
  aucun traceback et le code de sortie du processus n'a pas été capturé. Le point est signalé comme
  tel, pas comme un fait.
* Le critère (d) est mesuré sur un run dont le batch a changé en cours de route : il est **manqué**
  au sens de la consigne, mais il n'est pas démontré que la configuration cible (texte, batch 16)
  échouerait, elle aussi.
* L'ETA du run texte est une **extrapolation** à partir de 5 839 arêtes ; elle n'a aucune valeur de
  mesure.
* Rien n'est dit ici sur la qualité linguistique du modèle : aucun run texte n'a eu lieu, aucun
  échantillon n'a été généré, et ppl ≈ 1,0 sur le corpus toy ne mesure que la convergence du jouet.
