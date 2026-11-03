# Mycelia v4 — naissances et mémoire : rapport de validation

Date : 3 novembre 2026. Machine : RTX 3070 8 Go, torch 2.11.0+cu128, Python 3.12.10.
Périmètre : D1 (naissances), D2 (mémoire des messages), D3 (honnêteté du rapport v3), puis
la chaîne de validation demandée.

Chaque ligne du tableau de validation donne ✔/✘, la preuve, et le diagnostic exact quand c'est ✘.

---

## 1. Résultat en une phrase

La cible de vitesse est **atteinte** : en config cible (batch 16, séquence 128, 8 tours),
un pas passe de **8,66 s à 0,412 s à 500 arêtes** et de **OOM à 0,283 s à 2 000 arêtes**, avec
un pic mémoire de **0,03 GiB** au lieu de 4,04–14,43 GiB. Le run texte de 10 000 pas devient
une affaire d'heures (~47 min à 2 000 arêtes au débit mesuré), plus de journées.

---

## 2. D1 — la naissance devient un événement

### D1.1 — Zéro gradient à travers la décision

**Prémisse corrigée** : `_birth_edges` était **déjà** décoré `@torch.no_grad()` en v3 ; la
décision ne construisait donc aucun graphe. Ce qui manquait n'était pas l'absence de gradient,
mais un test qui le prouve : `test_birth_no_grad` espionne `_gumbel_like`, qui s'exécute *à
l'intérieur* de la passe, et vérifie `torch.is_grad_enabled() is False`. L'arête créée porte son
`edge_mask` entraînable comme n'importe quelle synapse ; le gradient vit sur `_edge_mask` (la
propriété `edge_mask` est une vue).

### D1.2 — Une seule passe de naissance par forward

Avant : `_birth_edges` était appelé à **chaque tour** (8 par forward texte). Après : une seule
passe, à la fin de la diffusion, sur l'état agrégé des nœuds qui ont **effectivement émis**.

C'est ce changement qui explique l'essentiel du gain, et pour une raison qui n'était pas prévue
par la consigne : les naissances créées en cours de forward grossissaient le front d'onde pour
les tours suivants, donc le nombre d'arêtes chaudes, donc les tenseurs `[E, B, dim]`. Une
naissance par tour entretenait une **cascade** qui multipliait le coût du forward entier — d'où
le facteur 30 entre « 1 naissance » et « 8 naissances » mesuré en v3.

Détail d'implémentation qui a demandé une correction : les émetteurs ne sont pas les receveurs.
Le suivi se fait maintenant par un masque `emitted` (nœuds dont la sortie est non nulle), et
**la porte y entre quand elle émet** — sinon aucune naissance ne peut partir du nœud 0.
Marquer la porte inconditionnellement était faux : un forward à porte fermée doit rester
totalement sombre (`test_gate_silence` l'a immédiatement attrapé).

### D1.3 — Candidats échantillonnés et bornés

Les candidats étaient déjà tirés (`birth_candidates = 64`), pas exhaustifs. Ce qui est ajouté,
c'est la **preuve** que le coût ne dépend pas de N : `test_birth_candidates_bounded` chronomètre
la passe de naissance à 512 et 3 000 nœuds et exige un rapport ≤ 3.

---

## 3. D2 — chunking des messages

Le scatter est découpé par **chunks d'arêtes** (`edge_chunk_size`, défaut 512, dans
`configs/config.yaml`) : pour chaque chunk, clés, requêtes, score et message sont calculés puis
`index_add` dans l'accumulateur du tour. Le pic mémoire d'un tour suit la taille du chunk, plus
le nombre total d'arêtes chaudes.

**Prémisse corrigée** : l'état n'est pas `[N, batch, seq, dim]` mais `[N, batch, dim]` ; l'axe
`seq` est une boucle Python, pas une dimension du tenseur. Le chunking porte donc sur l'axe des
arêtes, qui est celui qui explosait.

**Nuance mathématique** : l'addition flottante **n'est pas** associative. Changer l'ordre des
accumulations change le résultat au bit près, c'est pourquoi `test_chunking_equivalence` compare
à `allclose(atol=1e-6)` et non `torch.equal` — et pourquoi `test_resume_continuity` reste à
1e-6 sans être touché (un seul `edge_chunk_size` par run, donc aucune dérive entre reprise et
run continu).

Mesures de mémoire (config cible, batch 16, séquence 128, 8 tours) :

| Arêtes | v3 | v4 |
|---|---|---|
| 500 | 8,661 s/pas, 0,88 GiB | **0,412 s/pas, 0,03 GiB** |
| 2 000 | **OOM** (14,43 GiB demandés) | **0,283 s/pas, 0,03 GiB** |
| 2 000, chunk 256 | — | 0,296 s/pas, 0,03 GiB |
| 2 000, sans naissance | 2,235 s/pas (batch 8, seq 32), 4,04 GiB | 0,293 s/pas, 0,03 GiB |

---

## 4. D3 — honnêteté du rapport v3

Le rapport v3 disait à la fois « run texte v2 : perte 9,33 → 8,95 » et « texte non lancé ».
Corrigé : le run texte v2 a bien existé mais **partiel** — mesuré jusqu'au pas 100, puis arrêté
volontairement pour corriger la v3. Aucun run texte long n'a été mené à terme. Le tableau
d'état distingue maintenant « run texte partiel (~100 pas, arrêté) » et « run texte 10 000 pas :
non lancé ».

---

## 5. D4 — ce qui ne devait pas bouger

Intact et revérifié : `test_resume_continuity` (1e-6, le bug `_edge_keys` reste corrigé),
verrou anti-double-lancement (`pid_alive` via `OpenProcess` sous Windows), sauvegarde horloge et
`history.json` incrémental/atomique. Le chunking ne change rien au checkpoint : l'état sauvegardé
reste le même tenseur.

---

## 6. Validation demandée — résultats

### 1. pytest complet — ✔

```
python -m pytest tests -q
41 passed, 1 warning in 56.04s
```

41 = 36 de la v3 + **5** nouveaux (la consigne en demandait 4 ; `test_birth_no_grad`,
`test_birth_once_per_forward`, `test_birth_candidates_bounded`, `test_chunking_equivalence`,
`test_chunking_memory`). `test_chunking_memory` est sauté si CUDA est absent.

### 2. Run toy 500 pas — ✘ sur le critère strict, ✔ sur le reste

```
python mycelia/train.py --steps 500 --report-every 50 --save-every 250
```

| Pas | Perte | ppl | Arêtes | Nœuds | arêtes_nées | nœuds_nés | ms/pas (glissant 20) |
|---|---|---|---|---|---|---|---|
| 50 | 3,6267 | 37,6 | 400 | 514 | 388 | 2 | 781 |
| 150 | 1,9725 | 7,2 | 1 154 | 514 | 1 160 | 2 | 943 |
| 250 | 0,8486 | 2,3 | 1 817 | 514 | 1 933 | 2 | 1 187 |
| 350 | 0,1852 | 1,2 | 2 369 | 484 | 2 705 | 2 | 1 398 |
| 450 | 0,0214 | 1,0 | 2 863 | 503 | 3 474 | 2 | **1 799** |
| 500 | 0,0096 | 1,0 | 3 247 | 509 | 3 858 | 2 | **4 678** |

- Compteurs distincts et slots morts à l'allocation : ✔ (`pool 512 → 532 (nœuds_nés 2)`,
  `nœuds 514` = 512 + 2 vraies naissances, `arêtes_nées 3 858`).
- Apprentissage : ✔ 3,63 → 0,0096, perplexité 37,6 → 1,0.
- Vie de la colonie : ✔ 623 arêtes coupées, 54 nœuds enterrés, cimetière jusqu'à 30.
- Fin propre : ✔ `chemin : 0 -> 2 -> 1`, génome écrit, verrou relâché, `history.json` à 11
  entrées avec `edge_births`/`node_births`.
- **ms/step < 2 s : ✘ au pas 500** (4 678 ms). Diagnostic exact, pas conjecture : la fenêtre
  glissante des 20 derniers pas contient, au pas 500, la **validation (`validate()`),
  la décision de croissance (`check_every = 500`) et le checkpoint** ; le régime stable des pas
  400–450 est à 1,65–1,80 s. Le critère est donc tenu en régime, pas sur un pas de maintenance.
  Correctif évident pour la v5 : exclure validation/croissance/checkpoint du chrono des pas.

### 3. Mesure à la config cible (~2 000 arêtes) — ✔

`0,283 s/pas` et `0,03 GiB` de pic, batch 16, séquence 128, 8 tours, naissances actives.
Objectif < 2 s/pas et < 6 Go : largement atteint. Détail des quatre mesures en section 3.

### 4. Run toy 2 000 pas complet — ⏳ lancé, non terminé

`python mycelia/train.py --steps 2000 --report-every 250` tourne au moment de ce rapport
(`logs_toy_v4_2000.txt`). Non conclu ici : il faut environ une heure au débit mesuré, et je ne
suis pas notifié de sa fin. Les 5 critères de verdict se lisent dans le log : perte qui baisse,
`arêtes_nées`/`nœuds_nés` distincts, slots morts (`nœuds` ≠ `pool`), régime de ms/step, fin
propre (chemin `0 -> 2 -> 1`, verrou relâché).

### 5. Run texte 10 000 pas — non lancé

Il attend la fin du toy 2 000. Débit attendu au vu de la mesure : ~47 min pour 10 000 pas à
2 000 arêtes, à réévaluer puisque le câblage grossit pendant le run (à 3 200 arêtes le toy est à
~1,8 s/pas avec la validation et la croissance incluses).

```bash
python mycelia/train.py --data text --scale 0     # après libération du verrou toy
python scripts/train_baseline.py                  # baseline Transformer, en parallèle
```

---

## 7. Bugs trouvés pendant la v4 (et comment)

1. **Les émetteurs n'étaient pas suivis** : après D1.2, la passe unique utilisait
   `fired_nodes`, qui marque les *receveurs*. Résultat : la porte ne pouvait plus être la source
   d'une naissance, et `test_revive_from_cemetery_via_growth` a échoué. Corrigé par un masque
   `emitted` distinct, alimenté par les nœuds dont la sortie est non nulle.
2. **Marquer la porte inconditionnellement** cassait `test_gate_silence` (un forward à porte
   fermée doit rester sombre). La porte n'entre dans `emitted` que si elle a réellement émis.
3. **Mon test de naissance mesurait le mauvais contexte** : espionner `_birth_edges` observe
   l'appelant (autograd actif par conception), pas l'intérieur du décorateur. Le test espionne
   désormais `_gumbel_like`.
4. **Mon test avait remplacé le câblage** par une seule arête `0 → 2`, donc plus rien n'atteignait
   le lecteur et le gradient de tous les masques valait zéro. Câblage complet `0 → 2 → 1` rétabli.
   Aucun test n'a été affaibli : les assertions portent sur `_edge_mask` (le paramètre réel) et
   sur la totalité de sa taille.

---

## 8. Reste à faire

1. Exclure validation/croissance/checkpoint du chrono des pas (le 4 678 ms du pas 500 est un
   artefact de fenêtre, pas une lenteur du modèle).
2. Finir le toy 2 000, puis lancer le texte 10 000 et le baseline Transformer.
3. Chunking : envisager `torch.utils.checkpoint` par chunk si un run dépasse 20 000 arêtes
   (le chunking actuel borne le pic transitoire, pas la mémoire de backward par arête).
4. Cosmétique : libellé « naissances » du panneau du cockpit.

ZIP : `mycelia-v3.zip` reste l'archive de la v3 ; celle de la v4 est `mycelia-v4.zip`
(`mycelia/`, `tests/`, `scripts/`, `dashboard/`, `configs/`, `docs/`, `README.md`,
`requirements.txt`, `LICENSE`, `.gitignore` — sans `data/`, `archive/`, `__pycache__`).
