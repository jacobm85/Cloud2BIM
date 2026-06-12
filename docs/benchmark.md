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

## Resultat 2026-06-12 — v4 (evidensraster) + brusläge

Bänken har nu `--noisy` (15 % felmärkta punkter + 6 skuggsektorer),
flervåningsscenariot `kontor2v`, v1-rader och `--fast` (hoppar v1).

Väggar F1, **brusigt läge** (det fältrelevanta):

| Scenario  | v4    | ml   | v2   | v1*  |
|-----------|-------|------|------|------|
| kontor    | 0.96  | 0.96 | 0.69 | 0.56 |
| bostad    | 0.86  | 0.93 | 0.88 | 0.63 |
| vård      | 0.76  | 0.91 | —    | —    |
| industri  | **0.92** | 0.36 | 0.75 | 0.18 |
| garage    | **0.89** | 0.57 | —    | —    |
| kontor30° | 0.85  | 0.89 | —    | —    |

\* v1 tar 3–120 **minuter** per scenario (övriga 0,1–1 s); v1-siffror
från körning med något äldre v4-kod, v1-raderna opåverkade.

- v4 slabbar: rätt antal nivåer i ALLA scenarier, clean och noisy,
  inkl. kontor2v (3 slabbar, 2/2 golvnivåer).
- v4 slår v1 överallt, med brus och utan — användarens fältobservation
  "v1 bäst" förklaras av att ML-vägen (inte v1) var trasig på riktiga
  skanningar; v4 ger histogram-robusthet + ML-precision.
- Verklig skanning (gråskala, 7,8M pkt): v4 hittar golv −0,94 m,
  takzon 1,71–2,15 m som EN slab (undertak+installation+stomme),
  13 väggar med trovärdiga tjocklekar på 1 s. Se temp/ (ej i git).

**Uppdatering 2026-06-12 em (b7b8d4d)** — ocklusionsmedveten persistens
(occuperade/tillgängliga skivor i stället för absolut andel) + bästa-
vägg-projektion för etikettkluster + portmerge-toleranser:

| Noisy | kontor | bostad | vård | industri | garage | kontor30 |
|-------|--------|--------|------|----------|--------|----------|
| väggar F1 | 0.96 | 0.86 | 0.78 | 0.92 | 0.89 | **0.92** |
| öppn. R   | 0.75 | 0.29 | **0.79** | **0.67** | 0.00 | 0.44 |

**Blekinge-validering** (287M pkt flervåningsbyggnad, ingen ML, ingen
GPU): korrekt nivåstruktur — golv (täckning 0,77) + trippelskiktad
takzon (undertak 7,40/7,50 + stomme 7,66) grupperad till EN slab;
205 väggar + 66 öppningar (24 dörrar) på 143 s. Tydlig korridor- och
rumsstruktur i preview (temp/blekinge_v4_preview.png).

**Kvarvarande v4-svagheter** (fortsättning på task #9):
1. Garage-porten under extremskugga: 0 öppningar (väggen förblir
   delad; syntetskuggorna är hårdare än multi-skanner-verklighet).
2. bostad noisy öppningar R=0,29 — små etikettkluster försvinner i
   skugga (datagräns snarare än algoritmfel).
3. Blekinge: vissa väggtjocklekar ser för feta ut i preview —
   verifiera tjockleksskattningen mot kända väggar.
4. Prestanda storvåning: 139 s (DBSCAN + 205 fasadraster) — profilera.

## Saknas ännu i bänken

Trappor, runda pelare, krökta väggar, varierande punkttäthet
(närfält/fjärrfält), multi-skanner-registreringsfel.
