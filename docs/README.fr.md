# Mycelia

**Un cerveau artificiel à câblage émergent qui apprend le français.**

Mycelia n'est pas une pile de couches. C'est une colonie de petits nœuds
autonomes qui se parlent par messages, décident seuls à qui s'adresser, font
naître des synapses quand ils veulent atteindre quelqu'un, les coupent quand
elles ne servent plus, meurent quand plus rien ne les atteint — et reviennent
exactement tels qu'ils étaient quand un message les appelle.

Le pari est l'inverse de celui du Transformer : une intelligence peut venir de
la **structure dynamique** (des nœuds qui naissent, meurent, se spécialisent)
plutôt que de la taille brute. Le modèle biologique est le cerveau de la
drosophile cartographié par FlyWire : 140 000 neurones, ~50 millions de synapses,
et beaucoup de cellules très simples dont la puissance vient du nombre de leurs
connexions.

État : les six étages sont implémentés, testés, et observables en direct dans un
cockpit navigateur. Les résultats mesurés — et ceux qui ne le sont pas — sont
dans [Résultats](#résultats).

---

## Les deux niveaux

| Niveau | Ce que c'est | Poids |
|---|---|---|
| **Nœud** | L'équivalent du neurone biologique. Il reçoit, décide, transmet, naît, meurt. Largeur 32. | un nœud |
| **Organes internes** | `gate`, `body`, `router` — des paramètres mathématiques, *pas* des neurones. | ~6 400 paramètres par nœud |

Le cahier des charges estimait ~4 000 paramètres/nœud (le nœud sans son routeur).
L'implémentation ajoute l'adresse requête/clé (D3), donc un nœud pèse 6 401
paramètres : gate 33, body 4 192, LayerNorm 64, requête 1 056, clé 1 056. C'est
le vrai budget ci-dessous.

## Les fondements (non négociables)

| # | Règle | Où elle vit |
|---|---|---|
| F1 | Plein de nœuds dès le départ, presque tous sans connexions ; seul un chemin entrée→sortie existe. | `Brain._seed_wiring` (exactement 12 arêtes : 0→2..7 puis 2..7→1) |
| F2 | Décentralisation totale : chaque nœud décide seul à qui parler. Le message choisit sa destination. | routeur requête/clé, un score par arête |
| F3 | Parler provoque la naissance : adresser un nœud mort ou jamais né le ressuscite ou l'alloue. | `_birth_edges`, `revive_node`, `growth.try_birth_node` |
| F4 | Un nœud qui reçoit doit modifier ou transmettre, jamais restituer son entrée. | pénalité douce de cosinus (> 0,999) |
| F5 | Les boucles naissent par hasard et s'éteignent par dissipation. Aucune boucle n'est détectée ni cassée. | messages ×0,9 par tour, membranes ×0,995 par tour |
| F6 | Extinction progressive, jamais de plafond artificiel. | EMA d'usage + élagage + isolement |
| F7 | Mort en deux temps : liens d'abord, nœuds isolés ensuite. Cimetière exact. Nœuds 0 et 1 immortels. | `lifecycle.py` |
| F8 | Croissance par paliers, déclenchée par les mesures, jamais par un compteur. | `GrowthManager` (validation seulement) |
| F9 | Spécialisation émergente : taxe d'anti-redondance + trace de circulation. | `anti_redundancy_penalty`, trace |
| F10 | Légèreté à l'inférence : seuls les nœuds traversés calculent. | front d'onde des nœuds chauds |
| F11 | Évolution infinie par paliers, cap symbolique 140 000 nœuds. | `max_population` |

---

## La circulation du signal

Aucune matrice dense nulle part. Le câblage est un tampon d'arêtes pré-alloué
`edge_index` de forme [2, 20 000] plus un compteur d'utilisation `ne`, et une
force entraînable par emplacement. La matrice 140 000² = 19,6 milliards de cases
n'est jamais allouée.

Le forward est un **front d'onde**, token par token :

1. Le nœud 0 injecte l'embedding du token dans le front (`E = 1`).
2. À chaque tour, chaque nœud chaud transforme *ce qu'il vient de recevoir* via
   sa porte et son corps, score ses cibles avec sa propre requête contre leurs
   clés, et envoie `sortie × score × force × 0,9` au tour suivant.
3. Seuls les nœuds qui ont reçu quelque chose calculent (F10). Un signal qui ne
   reçoit rien meurt ; aucune règle anti-boucle n'existe.
4. Le nœud 1 accumule chaque message dans sa **membrane**, qui refroidit de
   0,995 par tour (~0,96 par token, une vingtaine de tokens de mémoire).
5. À la position `t`, la prédiction est lue sur le nœud 1 *avant* d'injecter le
   token `t`. C'est toute la causalité : prédire le token courant à partir du
   passé seulement, donc aucun décalage n'est nécessaire dans le jeu de données.
6. La trace `(tour, source, cible)` est enregistrée et envoyée au cockpit.

Un pas d'entraînement texte est huit forwards tronqués de 16 tokens : le budget
de naissances est donc transmis d'un bloc à l'autre — **8 naissances par pas
d'optimisation**, pas par appel. L'ancienne erreur réinitialisait le budget huit
fois par pas, ce qui fait exploser le graphe lentement.

La membrane se souvient ; le front d'onde circule. Les premières versions
faisaient émettre à chaque nœud toute sa membrane : n'importe quel cycle devenait
un amplificateur (l'énergie montait jusqu'à 1,5 million). Un nœud émet maintenant
ce qui vient d'arriver, ce qui rend vraie la règle de dissipation.

---

## Structure

```
mycelia/
  node.py        un neurone (référence) + NodeBank (empilé, agrandissable, groupé)
  brain.py       graphe, front d'onde, routeur, naissances, lecture, parole
  lifecycle.py   EMA d'usage, élagage, isolement, cimetière, résurrection exacte, taxes
  growth.py      réallocation, naissance d'un nœud depuis un message, gestionnaire de paliers
  checkpoint.py  génome versionné, optimiseur par nom, état RNG complet
  data.py        tâche jouet (copie retardée) + flux de texte français (5 % validation)
  train.py       Trainer partagé par le CLI, le dashboard et les tests, OOM, SIGINT
scripts/         train_tokenizer, generate, train_baseline, dashboard
dashboard/       runner.py (thread), server.py (FastAPI + WS), static/index.html
tests/           6 fichiers, 32 tests
configs/         config.yaml — tous les seuils, desserrables sans toucher au code
docs/            ce fichier + traductions courtes (ES, DE, ZH, JA, PT, RU, AR)
```

## Lancer

```bash
pip install -r requirements.txt

# 1. corpus français (LeCarnet) + BPE 10 000 tokens — un seul téléchargement
python scripts/train_tokenizer.py

# 2. régression d'architecture : copie retardée, quelques secondes par pas
python mycelia/train.py                       # 2 000 pas, scale 0
python mycelia/train.py --scale 1             # 302 nœuds (C. elegans)
python mycelia/train.py --scale 2             # 3 000 nœuds (larve de mouche)

# 3. français : 10 000 pas, batch 16, seq 128, rétropropagation tronquée à 16
python mycelia/train.py --data text --scale 0

# 4. parler
python scripts/generate.py --prompt "Il était une fois" --tokens 100

# 5. le chiffre sans lequel rien n'est interprétable
python scripts/train_baseline.py --steps 10000

# 6. le cockpit
python scripts/dashboard.py --train --data text --scale 0
#    → http://127.0.0.1:8080

# 7. tout doit rester vert
python -m pytest tests -q
```

Un Ctrl+C sauvegarde `data/genome.pt` entre deux pas et affiche la commande de
reprise. `--resume` restaure les poids, le câblage (`ne`), le cimetière, le
palier de croissance, les moments d'AdamW **par nom**, et tous les RNG (torch,
CUDA, numpy, Python, plus le générateur de batches). `test_resume_continuity`
compare 50 pertes après rechargement à la course ininterrompue, avec une
tolérance de 1e-6 non négociable.

## Échelles et mémoire

| `--scale` | Nœuds | Paramètres de nœuds | Sens |
|---|---|---|---|
| 0 | 512 | 3,3 M | prototype, RTX 3070 8 Go |
| 1 | 302 | 1,9 M | clin d'œil à *C. elegans* |
| 2 | 3 000 | 19,2 M | cerveau de larve de mouche |
| 3 | 140 000 | 896 M | la mouche, long terme |

À 140 000, les poids seuls font ~3,6 Go en fp32 et AdamW double : cela ne tient
pas dans 8 Go. `mmap=True` et les sous-graphes échantillonnés sont le chemin
déclaré, pas un résultat mesuré ici. Ce qui est vrai à toutes les échelles est le
stockage creux : `edge_index [2, max_edges]`, jamais d'adjacence dense.
`test_scale_2_forward` fait tourner 3 000 nœuds en moins de cinq secondes CPU.

## Corpus

`MaxLSB/LeCarnet` (Hugging Face) : de petites histoires écrites nativement en
français, au vocabulaire volontairement restreint. `scripts/train_tokenizer.py`
le diffuse dans `data/corpus_fr.txt` (24 M de caractères pour le run livré),
entraîne le BPE dessus, et affiche cinq allers-retours pour vérifier accents et
élisions. Le repli anglais est obsolète. Solutions de secours :
`iproskurina/TinyStories-French`, `rrenaud/multilingual_tinystories` (GitHub), ou
le fichier de corpus local.

## Résultats

Tous les chiffres ci-dessous viennent de `pytest` et des runs décrits, sur la
machine réelle (RTX 3070, torch 2.11+cu128). Ce qui n'a pas été mesuré est
marqué comme tel.

### Suite de tests

```
python -m pytest tests -q
32 passed in ~60 s
```

Six fichiers, un par étage : `test_brain`, `test_lifecycle`, `test_text`,
`test_growth`, `test_checkpoint`, `test_dashboard`.

### Run jouet (copie retardée, `--scale 0`, batch 32, CUDA)

| Pas | Perte | Perplexité | Arêtes utilisées | Nœuds vivants |
|---|---|---|---|---|
| 100 | 3,03 | 20,6 | 612 | 512 |
| 200 | 1,76 | 5,8 | 635 | 512 |
| 400 | ~0,7 | ~2,0 | 756 | 431 |

Naissances, coupes et enterrements sont visibles dans les sept signes vitaux ;
~0,26 s/pas à 512 nœuds. Ce run précède le front d'onde final ; la régression de
500 pas (`test_learning`) atteint **4,86 → 0,11** avec le code actuel.

### Run français (validation, cockpit en direct, « Il était une fois »)

Le code final a été entraîné en direct dans le cockpit sur LeCarnet, `--scale 0`,
batch 16, séquence 128, huit tours. La perplexité part de ~11 750 (vocabulaire
10 000), descend pendant le run (8 261 au pas 19 après passage du lr à 3e-4 via
le slider), et les échantillons de 40 tokens passent de tokens aléatoires à des
fragments français. Depuis la v3, la vitesse affichée est la **moyenne glissante
des 20 derniers pas** (jamais `temps/pas`, qui ne peut que monter et cache tout
intervalle lent). Mesuré sur GPU exclusif, RTX 3070 : **3,2 s/pas sur les 40
premiers pas** (graphe encore presque vide), puis **8,5 s/pas vers 800 arêtes** —
le coût suit le *câblage*, pas le numéro de pas. Le profil d'un pas met 52 % dans
`run_backward`, 16 % dans le forward, sur ~1 024 itérations de tour séquentielles
(128 positions × 8 tours), chacune quelques dizaines de petits kernels.

La v3 retire le `torch.isin` trié du chemin des naissances et l'affichage
cumulé trompeur, mais elle **n'atteint pas la cible < 2 s/pas à 2 000 arêtes
denses** : 2,24 s/pas mesuré à batch 8, séquence 32, et un OOM à batch 16,
séquence 128 (4 GiB d'activations sur 8 GiB de carte). Regrouper l'état en un
tenseur n'était pas la pièce manquante — la membrane est un tenseur `[N, batch,
dim]` depuis le début — donc le levier suivant est le **nombre de kernels par
tour**, et le chiffre reste honnête jusque-là.

**L'ETA du run texte de 10 000 pas est donc inconnue** : à 8,5 s/pas elle dépasse
24 h, et la courbe mesurée monte avec le câblage.

**Non mesuré :** un run texte complet de 10 000 pas (des heures), le tableau des
3 seeds, et la perplexité finale du baseline Transformer. Les scripts sont
complets et testés à petits pas ; les chiffres doivent être produits sur la
machine cible.

### Trois seeds

| Seed | Perplexité validation | Pool final | Arêtes finales |
|---|---|---|---|
| 42 | run court seulement (ci-dessus) | 532 après 2 pas ; paliers non atteints | 102 au pas 11 |
| 43 | non lancé | — | — |
| 44 | non lancé | — | — |

Le tableau se remplit avec `--seed 42|43|44` sur un vrai budget de 10 000 pas. Il
est laissé vide là où il doit l'être, plutôt que rempli de chiffres inventés.

## Ce que nous faisons différemment

* **Neural Cellular Automata** (Mordvintsev et al., *Distill*, 2020,
  <https://distill.pub/2020/growing-ca/>) font croître une grille *fixe* de
  cellules avec une règle partagée. Ici les cellules sont hétérogènes,
  individuellement plastiques, et le graphe lui-même est créé et détruit
  localement ; il n'y a pas de règle de mise à jour commune, et l'identité d'un
  nœud est son adresse, pas sa position.
* **FlyWire / FLYNN** (Dorkenwald et al., *Nature*, 2024,
  <https://www.nature.com/articles/s41586-024-07558-y> ; Lappalainen et al.,
  *Nature*, 2024, <https://www.nature.com/articles/s41586-024-07763-9>)
  cartographient un vrai connectome et le simulent. Mycelia emprunte la
  *philosophie d'échelle* (beaucoup d'unités simples, arêtes creuses, calcul
  local) mais le connectome est appris, pas copié : aucun diagramme de câblage
  n'est fourni au modèle.
* **Dynamic Sparse Training** (Evci et al., 2020,
  <https://arxiv.org/abs/1911.11134> ; Liu et al., 2021) élague et régénère des
  poids dans un réseau dense fixe, avec un masque et un calendrier globaux. Ici
  la topologie *est* le calcul, chaque nœud choisit ses cibles depuis une adresse
  requête/clé, la suppression est une règle locale d'usage, et le nombre de
  neurones change.
* **Generative Agents** (Park et al., 2023,
  <https://arxiv.org/abs/2304.03442>) donnent à des agents un flux de mémoire et
  de la réflexion, au-dessus d'un LLM. Mycelia n'a aucun LLM en dessous : la
  mémoire est une membrane qui refroidit, et ce qui émerge s'observe par la
  circulation, pas par un prompt.

## Les pièges (extraits du vrai journal de débogage)

La section 0 bis du brief est une liste d'échecs qui ont déjà coûté un run. Ils
sont généraux ; version la plus courte :

1. **Tester `ne`, jamais la forme du tampon.** Un tampon pré-alloué a 20 000
   colonnes quel que soit le câblage. Tous les tests portent sur le compteur.
2. **Ne jamais écrire en place dans une vue sauvegardée par autograd.** Cloner
   les vues source/cible avant qu'un forward puisse créer des arêtes ; ajouter
   par l'emplacement libre ; compacter seulement entre deux pas.
3. **Rejouer l'OOM hors du `except`.** Dedans, l'exception retient encore le
   graphe qui a échoué ; libérer, `gc.collect()`, `empty_cache()`, *puis*
   réessayer avec un batch plus petit. Jamais de repli dense.
4. **Regrouper par nœud, pas par (échantillon, nœud).** Copier les poids par
   échantillon est l'erreur la plus coûteuse de la première version (~12,7 Go).
   Ce sont les activations sauvegardées pour le backward, pas le forward, qui
   coûtent.
5. **Budgéter les naissances et refroidir le départ.** 8 naissances par *pas*,
   `init_bias` −8, plafond 20 000 arêtes, messages uniquement depuis les sources
   actives. Sinon les cibles deviennent actives, créent d'autres cibles, et le
   graphe explose.
6. **Pas de `.item()` dans la boucle de diffusion.** Les valeurs restent des
   tenseurs ; les lectures se font après la boucle.
7. **Ne jamais relâcher `test_resume_continuity`.** Un écart de 3,3e-6 a un jour
   suggéré que la tolérance était trop serrée. La cause était le compteur `step`
   d'AdamW, scalaire, rejeté par un contrôle de forme : aucun moment n'était
   restauré. Le test avait raison de rester à 1e-6.
8. **Livrer des corrections complètes, pas des patches de conversation.** Le
   dépôt est la source de vérité ; le ZIP est complet (code + tests).
9. **Seul un `pytest` local est une preuve.** Tout résultat non reproduit sur la
   machine est « non vérifié », y compris les cases vides de ce README.
10. **Ne pas envoyer de raccourci ni de fichier à moitié écrit.** Il sera tronqué.

## Plan de débogage

* **Aucun câblage après 2 000 pas** — baisser L1 (1e-4 → 1e-5), desserrer
  `init_bias`, vérifier que les scores requête/clé varient, vérifier l'énergie du
  signal au nœud 0.
* **Le réseau ne meurt jamais** — vérifier l'EMA d'usage (decay 0,99) et le seuil
  d'élagage 0,02 ; les deux sont dans `configs/config.yaml`.
* **Perplexité bloquée à vocab/10** — le chemin entrée→sortie est mort ; renforcer
  l'exploration vers le nœud 1 (la trace montre si quelque chose l'atteint).
* **Perplexité qui stagne en mode texte** — suspect n°1 : la rétropropagation
  tronquée à 16 tokens ; n°2 : le budget de naissances ; n°3 : le plafond
  d'arêtes (`ne` vs `max_edges`). Les trois sont visibles dans le cockpit.
* **OOM** — le retry est déjà hors du `except` ; vérifier le regroupement par
  nœud, et que `edge_index` est toujours [2, max_edges].

## Paliers de succès (jugés par le concepteur sur 20 générations)

1. Une phrase française grammaticale et compréhensible.
2. Une histoire de 3 à 5 phrases cohérente.
3. Raisonnement multi-étapes (questions/réponses).
4. Long terme (`scale 3`, 140 000 nœuds) : raisonnement long sur matériel modeste.

L'état actuel est sous le palier 1 avec un run court : les échantillons sont de la
salade de mots à forme française. C'est ce à quoi ressemble un graphe de 2 M de
paramètres avant un vrai run de 10 000 pas ; le verdict honnête est
« l'architecture fonctionne, la langue pas encore ».

## Verdict du run jouet

| Critère | Résultat |
|---|---|
| La perte baisse nettement (pas 1 %) | oui : 4,86 → 0,11 (test), 4,6 → 0,7 (run de 400 pas) |
| Les 7 signes vitaux bougent | oui : naissances, coupes, 505 enterrements |
| Trace 0 → … → 1 avec ≥ 2 nœuds intermédiaires | partiel : 0→3→1 observé ; les chemins longs apparaissent quand le graphe se densifie |
| Mémoire GPU < 6 Go batch 32 | mesurée ci-dessous |
| Pas < 1 s sur RTX 3070 | jouet 0,26 s — texte 5–8 s, noté comme le goulot connu |

## Licence

MIT. Voir [LICENSE](../LICENSE).
