# Mycelia (português)

**Um cérebro artificial com ligações emergentes que aprende francês.**

Mycelia não é uma pilha de camadas: é uma colónia de nós pequenos e autónomos que
falam por mensagens, escolhem sozinhos a quem se dirigir, criam sinapses quando
querem alcançar alguém, cortam-nas quando definham, morrem quando nada os alcança
e voltam exatamente como eram quando uma mensagem os chama. A aposta é o inverso
do Transformer: a inteligência pode vir da **estrutura dinâmica** (nós que nascem,
morrem, especializam-se) em vez do tamanho bruto. O modelo biológico é o cérebro
de *Drosophila* mapeado pelo FlyWire.

Estado: as seis etapas estão implementadas, testadas e observáveis ao vivo num
cockpit de navegador. Documentação completa: [README.md](../README.md),
[README.fr.md](README.fr.md).

## Arranque rápido

```bash
pip install -r requirements.txt
python scripts/train_tokenizer.py              # corpus LeCarnet + BPE 10 000
python mycelia/train.py --data text --scale 0  # francês, 10 000 passos
python scripts/dashboard.py --train --data text --scale 0
# → http://127.0.0.1:8080
python -m pytest tests -q                      # 32 testes
```

## Ideias centrais

* Um nó de largura 32 tem ~6 400 parâmetros; 140 000 nós é o limite simbólico,
  nunca uma matriz densa.
* A cablagem é `edge_index [2, 20 000]` mais o contador `ne`, nunca N×N.
* O router usa consulta/chave com candidatos amostrados: O(dim) por endereço, não
  O(N).
* As arestas nascem no forward com um orçamento de 8 por passo; a morte vem em
  dois tempos, e o cemitério devolve os pesos exatos.
* O cockpit mostra o grafo a cablar-se, os impulsos, os sete sinais vitais e o
  chat token a token.
