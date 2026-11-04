# Mycelia — rapport de reprise du 5 novembre

Date : 5 novembre 2026, 00 h 35 (heure locale) — état arrêté au pas 700 du run toy.
Périmètre : `C:\Users\alexw\Documents\mycelia`, reprise après l'incident de suppression.
Machine : RTX 3070 Laptop, torch 2.11.0+cu128, Python 3.12.10.

Ce rapport ne rapporte que ce qui a été **mesuré ou constaté dans ce dossier**. Les cases non
vérifiées le disent explicitement. Il remplace, pour cette session, le `RAPPORT-COMPLET.md`
annoncé dans la consigne de reprise : ce fichier **n'existe pas** ici (voir §7).

---

## 1. Résultat en une phrase

Le dossier restauré contient **déjà la v4 finale** : format 3, 41 tests verts, C1–C8 et D1–D4
présents et vérifiés, dépôt git en place, données intactes. Aucune réapplication n'a donc été
faite ; le run toy 2 000 pas a été lancé et a atteint le pas 700 (perte 0,0023), mais il ralentit
nettement et sature la mémoire de la machine (§6). Ce ralentissement est **reproductible** : le run
v4 précédent montre la même falaise. Le run texte n'a donc pas été lancé.

---

## 2. Prémisse de la consigne et réalité du dossier

La consigne décrivait « un snapshot antérieur (format 1, 28 tests, sans verrou, sans chunking,
sans slots morts) ». Le constat est l'inverse :

| Point annoncé | Constat mesuré | Preuve |
|---|---|---|
| format 1 | **format 3** (`READABLE_VERSIONS = (2, 3)`) | `mycelia/checkpoint.py:35-36` |
| 28 tests | **41 tests** (8+4+6+11+7+5) | `pytest -q` → 41 passed |
| sans verrou | `pid_alive`, `acquire_lock`, `release_lock`, et verrou réellement exercé | `mycelia/train.py:81,113,137` |
| sans chunking | `edge_chunk_size: 512` + tests d'équivalence et de mémoire | `brain.py:632`, `configs/config.yaml:23` |
| sans slots morts | allocation en réserve morte (`torch.zeros` puis `alive[:old_n]`) | `brain.py:360-361` |

Aucun `FORMAT_VERSION = 1` n'existe dans le dossier (recherche sur tout le code hors `archive/`).
Les anciennes générations présentes sont `archive/prototype`, `archive/prototype-legacy`,
`archive/v2`, `archive/v2-1633`.

**Conséquence :** réappliquer C1–C8 puis D1–D4 sur ce code aurait dupliqué des correctifs
existants et créé des commits annonçant un travail non effectué. La réapplication a donc été
remplacée par une **vérification bloc par bloc**, conformément à l'esprit de la consigne
(« pas en aveugle »).

---

## 3. Vérification C1–C8

| Bloc | Attendu | Constat | Preuve |
|---|---|---|---|
| C1 | verrou anti-double-lancement | présent et **exercé en réel** : verrou périmé PID 8848 détecté puis remplacé par PID 15108 | `train.py:81,113,137` ; log `logs_toy_v4_2000_rerun.txt` |
| C2 | sauvegarde horloge + historique atomique | `save_seconds: 600`, `first_save_step: 10`, écriture `tmp` + `os.replace` | `train.py:161,296-297` ; `config.yaml:53-54` |
| C3 | slots alloués nés morts | `grow_population` crée `torch.zeros(new_n, bool)` et recopie seulement les vivants | `brain.py:360-367` |
| C4 | performance / pas de `torch.isin` chaud | chunking par arêtes, appartenance host-side `_edge_keys` | `brain.py:632` ; tests `test_chunking_*`, `test_scaling_edges` |
| C5 | compteurs distincts | `edge_births` = arêtes nées, `node_births` = vraies naissances, dans métriques, logs et `history.json` | `train.py:402-403` |
| C6 | croissance transparente | messages `phase CROISSANCE`, `phase OPTIMISATION`, `PALIER` | `growth.py:147,154,159` |
| C7 | honnêteté documentaire | `docs/RAPPORT-v3.md` et `docs/RAPPORT-v4.md` distinguent mesuré / non mesuré | fichiers présents |
| C8 | décodage propre à l'affichage | `clean_text` utilisé par `generate.py` et le chat du cockpit, jamais dans le modèle | `data.py:35`, `generate.py:81,89`, `runner.py:366` |

---

## 4. Vérification D1–D4

| Bloc | Attendu | Constat | Preuve |
|---|---|---|---|
| D1.1 | décision de naissance hors gradient | `_birth_edges` décoré `@torch.no_grad()` ; `test_birth_no_grad` espionne `_gumbel_like` et vérifie `torch.is_grad_enabled() is False` | `brain.py:430-431` ; `tests/test_growth.py:195-201` |
| D1.2 | une seule passe de naissance par forward | compteur `birth_passes`, masque `emitted`, commentaire D1.2 dans le forward | `brain.py:539,674-687` ; `test_birth_once_per_forward` |
| D1.3 | candidats échantillonnés et bornés | `birth_candidates: 64` ; coût indépendant de N | `test_birth_candidates_bounded` |
| D2 | chunking des messages | `edge_chunk_size: 512`, équivalence à `atol=1e-6` (jamais `torch.equal`) | `brain.py:632` ; `test_chunking_equivalence`, `test_chunking_memory` |
| D4 | invariants de checkpoint | `test_resume_continuity` compare 50 pertes à **1e-6** ; verrou, horloge et `history.json` incrémental/atomique intacts | `tests/test_checkpoint.py:97` |

Vérification globale : `python -m pytest tests -q` → **41 passed, 1 warning, 57,62 s, exit 0**.
Le warning est une dépréciation Starlette/httpx (`fastapi.testclient`), sans effet sur les tests.

---

## 5. Portes du plan de reprise

| Porte | État | Preuve |
|---|---|---|
| 1. git init + `.gitignore` + commit initial | **déjà fait** : 2 commits, 129 fichiers suivis, arbre de travail propre | `db1ab80` (initial), `a1413ee` (tests) |
| 2. C1–C8 réappliqués un par un | **sans objet** : déjà présents, vérifiés §3 | — |
| 3. D1–D4 réappliqués un par un | **sans objet** : déjà présents, vérifiés §4 | — |
| 4. 41 tests verts, continuité à 1e-6 | **franchi** : 41 passed | sortie pytest |
| 5. corpus + tokenizer disponibles | **franchi, sans reconstruction** : 24,0 M caractères, 29 402 lignes, vocab 10 000, `eos = 3`, aller-retour correct | `data/corpus_fr.txt`, `data/tokenizer.json` |
| 6. ZIP horodaté avant tout run long | **franchi** : `mycelia-v4-2026-11-04.zip`, 39 fichiers, 116 Ko, hashes identiques au disque | comparaison SHA-256 |
| 7. run toy 2 000 puis texte 10 000 | **toy lancé**, texte non lancé (conditionné à la lecture des critères du toy) | §6 |

Détail ZIP : l'archive `mycelia-v4.zip` d'origine est elle aussi identique au disque
(25 entrées de code/configuration/README comparées, 0 différence). Le nouveau ZIP horodaté
ajoute `docs/` complet, y compris les rapports.

---

## 6. Run toy 2 000 pas

Commande : `python mycelia/train.py --steps 2000 --report-every 100`
Log : `logs_toy_v4_2000_rerun.txt` — processus d'entraînement PID 15108 (vivant au dernier relevé).

| Pas | Perte | ppl | Arêtes | Nœuds | arêtes_nées | nœuds_nés | ms/pas (moy. 20) |
|---|---|---|---|---|---|---|---|
| 100 | 2,9086 | 18,3 | 792 | 514 | 780 | 2 | 921 |
| 200 | 1,4417 | 4,2 | 1 541 | 514 | 1 547 | 2 | 1 017 |
| 300 | 0,3724 | 1,45 | 2 207 | 514 | 2 314 | 2 | 1 248 |
| 400 | 0,0771 | 1,08 | 2 760 | 503 | 3 088 | 2 | 1 471 |
| 700 | 0,0023 | 1,00 | 4 058 | 515 | 5 355 | 4 | 4 611 |

Autres constats du log :

* verrou périmé PID 8848 détecté et remplacé (C1) ;
* `pool 512 → 532` pour **2 vraies naissances**, compteurs distincts (C3, C5) ;
* élagage : 18 arêtes au pas 100, 101 au pas 200, 221 au pas 300, 276 au pas 400 ;
  **51 nœuds enterrés** au pas 300 (grâce d'isolement écoulée), 8 de plus au pas 400 ;
* cimetière à 11 au pas 400 pour 51 enterrements cumulés : **40 nœuds ont été rappelés par un
  message** (F3) — ce n'est pas une perte, c'est la résurrection en action ;
* mémoire GPU observée : ~6,5 GiB, utilisation ~31 % au voisinage du pas 100 ;
* `history.json` : 8 entrées à la rédaction (100, 200, 300, 400, 500 + `hold`, 500, 600, 700) ;
* **falaise de performance, reproductible** : dans ce run, 1 799 → 4 611 ms/pas entre 2 863 et
  4 058 arêtes ; dans le run v4 précédent (`logs_toy_v4.txt`), 1 799 → 4 678 ms/pas entre 2 863 et
  3 247 arêtes. Même marche de ×2,6 au même voisinage (~3 200 arêtes, pas ~500) : ce n'est pas un
  accident de cette session. Cause non isolée ; deux facteurs coïncident — la croissance du
  câblage et la saturation mémoire ci-dessous — et un micro-run profilé est nécessaire pour
  trancher ;
* **saturation mémoire machine** : le processus engage 12 591 Mo de mémoire validée (WS 3 571 Mo)
  pour une limite de commit de 35,2 Go, dont il ne reste que 0,0–0,2 Go. Conséquences mesurées :
  `import torch` échoue (WinError 1455), `fork` de Git Bash échoue, PowerShell ne démarre plus son
  CLR. Toute vérification qui importe torch, pytest compris, est impossible tant que le run vit.

**Nuance de lecture :** la ligne de signes vitaux est construite **avant** le cycle de vie du
même pas. Les colonnes `morts-arêtes` / `morts-nœuds` / `cimetière` ont donc un pas de retard :
au pas 300, la ligne affiche `morts-arêtes 119` et `cimetière 0`, alors que le cycle de vie
vient de couper 221 arêtes et d'enterrer 51 nœuds. Ce n'est pas une incohérence de mesure,
c'est l'ordre d'exécution (`metrics` puis `apply_lifecycle`).

### Critères de verdict (étape 4 de la validation précédente)

| Critère | État | Commentaire |
|---|---|---|
| Perte qui baisse nettement | ✔ | 2,91 → 0,37 entre les pas 100 et 300 |
| `arêtes_nées` / `nœuds_nés` distincts | ✔ | 2 314 arêtes vs 2 naissances |
| Slots morts (`nœuds` ≠ `pool`) | ✔ | 514 vivants pour un pool de 532 |
| ms/pas en régime (< 2 000) | ✗ | 0,92 → 4,61 s ; falaise reproduite par le run v4, donc pas imputable au seul environnement, mais la saturation mémoire empêche d'en isoler la cause |
| Fin propre (`0 -> 2 -> 1`, verrou relâché, génome écrit) | ⏳ | run en cours (pas 700) |

---

## 7. Écarts constatés et décisions non prises

1. **`docs/RAPPORT-COMPLET.md` absent.** Recherche par nom de fichier et par contenu dans tout
   le dossier (hors `.git`) : aucune occurrence. Les registres disponibles sont
   `docs/RAPPORT-v3.md` et `docs/RAPPORT-v4.md` ; la vérification s'appuie sur eux et sur le code.
2. **Hygiène git — corrigé pendant cette session.** `data/train.lock` et `.dash.pid` étaient
   suivis ; ils sont désormais ignorés et retirés de l'index (`git rm --cached`), commit
   `558c038` « gitignore: runtime locks (verrous d'exécution, pas de contenu) ».
3. **ZIP horodaté — corrigé.** `mycelia-v4-2026-11-04.zip` est maintenant suivi (commit
   `d0d0957` « archive: ZIP horodaté post-reprise »), comme les ZIP v3/v4 précédents. Il précède
   les commits de documentation de cette session.
4. **Aucun commit de réapplication n'a été créé.** Les commits « C1-C2 », « C3 », etc. demandés
   par le plan n'auraient correspondance avec aucune modification réelle.

---

## 8. Ce qui n'est pas fait

* **Run texte 10 000 pas : non lancé, volontairement.** Le critère (d) « ms/pas en régime
  < 2 000 » échoue de façon reproductible (§6) ; la règle du plan est alors STOP et diagnostic,
  pas lancement. À 4,6 s/pas, il coûterait ~13 h et saturerait la machine.
* **Baseline Transformer : non lancée** pour la même raison (mémoire de validation épuisée).
* **Verdict final du toy : non conclu.** Le run était au pas 700 à la rédaction, 1 300 pas
  restants à ~4,6 s/pas, soit ~100 min ; je ne suis pas notifié de sa fin.
* **Cause de la falaise : non isolée.** À instruire en priorité, sur machine au repos.
* **Baseline Transformer : non mesurée.** `baseline_ppl.txt` absent.
* **Table 3 seeds : non remplie.**

---

## 9. Commandes utiles

```bash
python -m pytest tests -q                                  # 41 tests
tail -40 logs_toy_v4_2000_rerun.txt                        # suivi du toy
python -c "import json; print(json.load(open('history.json'))[-5:])"
python mycelia/train.py --data text --scale 0              # run texte 10 000 pas, après le toy
python scripts/train_baseline.py --steps 10000             # baseline Transformer
```

Le run toy écrit `data/genome.pt` et `data/genome_latest.pt` toutes les 10 minutes d'horloge,
et `history.json` à chaque rapport. Le verrou `data/train.lock` est retiré proprement en fin de
run ; s'il subsiste, son PID est vérifié avant tout nouveau lancement.

---

## 10. Conclusion

La reprise est **techniquement terminée avant d'avoir commencé** : le code restauré est la v4
finale, les 41 tests passent, les corrections C1–C8 et D1–D4 sont en place et vérifiées, les
données sont intactes et le ZIP horodaté existe. Le seul travail neuf de cette session est la
vérification bloc par bloc et le lancement du run toy 2 000 pas, dont les premiers relevés
confirment le comportement attendu (perte 2,91 → 0,08, naissances, enterrements et résurrections visibles,
compteurs distincts, verrou opérationnel).

Le prochain jalon reste la **validation longue** : finir le toy, lire ses cinq critères, puis
lancer le texte 10 000 pas et la baseline.
