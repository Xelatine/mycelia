# Mycelia (Deutsch)

**Ein künstliches Gehirn mit emergentem Verdrahten, das Französisch lernt.**

Mycelia ist kein Stapel von Schichten, sondern eine Kolonie kleiner autonomer
Knoten. Sie sprechen über Nachrichten, wählen selbst, wen sie adressieren, bilden
Synapsen, wenn sie jemanden erreichen wollen, verlieren sie, wenn sie verkümmern,
sterben, wenn sie niemand mehr erreicht — und kehren exakt so zurück, wie sie
waren, wenn eine Nachricht sie ruft. Die Wette ist die Umkehrung des
Transformers: Intelligenz kann aus **dynamischer Struktur** entstehen (Knoten,
die geboren werden, sterben, sich spezialisieren) statt aus roher Größe.
Biologisches Vorbild ist das von FlyWire kartierte *Drosophila*-Gehirn.

Stand: alle sechs Stufen sind implementiert, getestet und live im Browser-Cockpit
beobachtbar. Vollständige Dokumentation: [README.md](../README.md),
[README.fr.md](README.fr.md).

## Schnellstart

```bash
pip install -r requirements.txt
python scripts/train_tokenizer.py              # LeCarnet-Korpus + 10 000er-BPE
python mycelia/train.py --data text --scale 0  # Französisch, 10 000 Schritte
python scripts/dashboard.py --train --data text --scale 0
# → http://127.0.0.1:8080
python -m pytest tests -q                      # 32 Tests
```

## Kernideen

* Ein Knoten der Breite 32 hat ~6 400 Parameter; 140 000 Knoten sind die
  symbolische Grenze, niemals eine dichte Matrix.
* Verdrahtung ist `edge_index [2, 20 000]` plus Zähler `ne`, nie N×N.
* Der Router arbeitet mit Query/Key und gesampelten Kandidaten: O(dim) pro
  Adresse statt O(N).
* Kanten entstehen im Forward mit einem Budget von 8 pro Schritt; der Tod kommt
  in zwei Stufen, und der Friedhof gibt die exakten Gewichte zurück.
* Das Cockpit zeigt das wachsende Netz, Impulse, die sieben Lebenszeichen und den
  Chat Token für Token.
