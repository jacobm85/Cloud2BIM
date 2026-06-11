# Syntetisk benchmark — `scripts/synthbench.py`

Genererar punktmoln med facit (väggar/bjälklag/öppningar/pelare/klutter/
skannerbrus) och perfekta S3DIS-etiketter, kör detektorerna och mäter
precision/recall. Orakeletiketterna isolerar extraktionsalgoritmerna från
segmenteringsmodellens kvalitet.

```
.venv/Scripts/python.exe scripts/synthbench.py             # alla scenarier
.venv/Scripts/python.exe scripts/synthbench.py industri    # ett scenario
```

## Resultat 2026-06-11 (efter ML-omskrivningen, commit-serie 7a0728b→)

| Scenario  | walls[ml] F1 | axis_err | walls[v2] F1 | openings P/R | columns R |
|-----------|--------------|----------|--------------|--------------|-----------|
| kontor    | **1.00**     | 4.9 cm   | 0.75         | 1.00/1.00    | —         |
| bostad    | **1.00**     | 8.9 cm   | 0.88         | 1.00/0.86    | —         |
| vård      | **1.00**     | 4.9 cm   | 0.91         | 1.00/1.00    | —         |
| industri  | **1.00**     | 6.2 cm   | 0.92         | 1.00/1.00    | 12/12     |
| garage    | **1.00**     | 1.1 cm   | 0.89         | 1.00/1.00    | 9/9       |
| kontor30° | **1.00**     | 5.7 cm   | 0.86         | 1.00/1.00    | —         |

Anteckningar:

- `walls[vertical]` hittar **0** väggar i alla scenarier:
  `vertical_min_points_per_slice=5` kräver ~5 punkter per 5×5×5 cm cell,
  vilket motsvarar 2 000 pkt/m² — orealistiskt efter dilution. Detektorn
  behöver täthetsadaptiva trösklar eller utfasning.
- `walls[v2]` övergenererar (P 0.60–0.86) — falska väggar ur kluttret
  trots ML-filtret; klusterfragment promoteras till singletons.
- Kvarvarande `axis_err` på ML-vägen är väsentligen
  `singleton_thickness`-antagandet (30 cm) mot facits 40 cm-ytterväggar —
  okunskap, inte algoritmfel; tjockleken kan inte mätas från en sida.
- Buggen som hittades av bänken: open3d:s `cluster_dbscan(min_points=…)`
  är ett *kärnpunktskrav* (grannar inom eps), inte klusterstorlek — glesa
  fönster-/dörrpunkter bildade aldrig kluster ⇒ 0 öppningar. Fixad i
  `openings_ml.py` (lågt kärnkrav + storleksfilter efteråt).

## Saknas ännu i bänken

Flervåningsscenario (bjälklagssammanslagningen!), trappor, runda pelare,
krökta väggar, ofullständig skanning (skuggade väggpartier), etikettbrus
(simulera modellfel: x % felklassade punkter).
