# Mycelia (español)

**Un cerebro artificial con cableado emergente que aprende francés.**

Mycelia no es una pila de capas: es una colonia de nodos pequeños y autónomos que
se hablan mediante mensajes, eligen solos a quién dirigirse, crean sinapsis
cuando quieren alcanzar a alguien, las cortan cuando se mueren de hambre, mueren
cuando nada los alcanza y vuelven exactamente como eran cuando un mensaje los
llama. La apuesta es la contraria a la del Transformer: la inteligencia puede
venir de la **estructura dinámica** (nodos que nacen, mueren y se especializan)
más que del tamaño bruto. El modelo biológico es el cerebro de *Drosophila*
cartografiado por FlyWire.

Estado: las seis etapas están implementadas, probadas y visibles en directo en un
cockpit de navegador. Documentación completa: [README.md](../README.md) y
[README.fr.md](README.fr.md).

## Inicio rápido

```bash
pip install -r requirements.txt
python scripts/train_tokenizer.py              # corpus LeCarnet + BPE 10 000
python mycelia/train.py --data text --scale 0  # francés, 10 000 pasos
python scripts/dashboard.py --train --data text --scale 0
# → http://127.0.0.1:8080
python -m pytest tests -q                      # 32 pruebas
```

## Ideas clave

* Un nodo de ancho 32 tiene ~6 400 parámetros; 140 000 nodos son el límite
  simbólico, nunca una matriz densa.
* El cableado es `edge_index [2, 20 000]` + contador `ne`, jamás N×N.
* El enrutador es consulta/clave con candidatos muestreados: O(dim) por
  dirección, no O(N).
* Las aristas nacen en el forward con un presupuesto de 8 por paso; la muerte va
  en dos tiempos y el cementerio devuelve los pesos exactos.
* El cockpit muestra el grafo cableándose, las impulsos, los siete signos
  vitales y el chat token a token.
