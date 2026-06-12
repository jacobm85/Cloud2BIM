# v5 — total nybyggnad: pipeline, detektor och GUI

Direktiv (användaren, 2026-06-12): bygg om RUBBET från blankt blad —
pipeline, stegflöde och GUI. Behåll endast det som är cutting edge ur
det gamla. Målet är inte fler features utan **extrem tillförlitlighet**:

1. Väggar HITTAS (båda riktningarna, korta som långa).
2. Hörn blir exakt rätt.
3. Inga överlappande väggar.
4. Inga kedjor av små sneda segment där verkligheten har EN lång vägg.

## Varför arkitekturen måste bytas (inte bara detektorn)

- v4 förbättrade detektionen men ärvde pipelinens tysta felkultur:
  `web/main.py` nedgraderade tyst v4→v1 (ac92bac) så en hel dags
  fälttest körde fel algoritm utan att någon märkte det. I v5 är
  tysta fallbacks FÖRBJUDNA — fel ska skrika, inte vikas undan.
- Per-vägg-detektion kan aldrig garantera kraven 2–4: hörn, överlapp
  och fragmentering är *relationer mellan väggar*, och måste lösas
  globalt, inte efterstädas med snapp/merge-heuristik.
- Kravet "pålitlig utan ML" är dimensionerande: servern kör utan GPU.

## v5-kärnan: global planlösningsoptimering (cellkomplex)

Per våningsplan:

1. **Bärlinjer** (carriers): v4:s evidensraster (vertikal persistens +
   ocklusionsnormalisering — beprövat, behålls) → få, långa, raka
   linjehypoteser via Houghliknande ackumulering i de dominanta
   riktningarna. En carrier är OBEGRÄNSAT lång — fragment existerar
   inte på den här nivån.
2. **Arrangemang**: carriers skär varandra → planet delas i celler
   (polygoner). Hörn är nu skärningspunkter PER KONSTRUKTION — exakta,
   delade, utan glapp.
3. **Cellklassning**: varje cell får inne/ute/rum-status från
   golvtäckning + punkttäthet + (om ML finns) etikettandel som mjuk
   vikt. Global optimering (graph cut / girig energiminimering):
   datatermer per cell + släthet över cellgränser.
4. **Väggar = gränser** mellan celler med olika status (rum|rum,
   rum|ute) som har väggevidens på sig. En lång vägg är en lång
   gränssträcka på EN carrier — fragmentering omöjlig. Överlapp
   omöjligt (en gräns finns bara en gång i arrangemanget).
5. **Tjocklek/öppningar**: som v4 (fasparning, frånvarohål + etikett-
   union, geometri-över-etikett) — det är cutting edge som behålls.
6. **Möbelimmunitet**: arkivhyllor/bilar bildar carriers men deras
   cellgränser saknar rumsskiljande funktion (samma rum på båda sidor,
   ingen golvdiskontinuitet) → klassas bort GLOBALT, inte per segment.
   Detta är svaret på Blekinge-croppens väggröta.

## Pipeline v5 (nytt, minimalt)

```
ingest  →  levels  →  plan(per våning)  →  elements  →  ifc
```

- **ingest**: strömmande läsning → normaliserat arbetsformat
  (points.zarr/npz + meta). Dilution/denoise/crop HÄR, en gång.
  287M-punktsmolnet får ALDRIG läsas om i senare steg (serverns kör
  läste om E57:an i varje steg — 8 min × 9 steg).
- **levels**: v4:s täckningskvalificerade Z-toppar (behålls).
- **plan**: cellkomplexet ovan. Deterministisk, seedad.
- **elements**: väggar+öppningar ur planen; pelare/trappor som plugins.
- **ifc**: ren exportmodul (ifcopenshell-delen behålls).
- Ett konfigschema, EN algoritm (inga v1/v2/vertical/v4-val, ingen
  hybrid-dispatch). Legacy-koden lämnas orörd på sin gren men ingår
  inte i v5-flödet.

## GUI v5 (nytt, minimalt, granskningsfirst)

- En sida: punktmoln + detekterad plan SIDA VID SIDA per våning,
  synkad kamera. Granskning är huvudflödet, inte ett efterhängt steg.
- Tre reglage, inte trettio: våningsval, känslighet (en skala),
  byggnadstyp. All öppen parametrik bakom "Avancerat".
- Differensvy: "detta ändrades sedan förra körningen".
- Jobbkö som visar VAD som körs (algoritm, läge) — aldrig mer tyst
  nedgradering.

## Verifiering

- synthbench behålls och utökas; blekinge_crop (temp/
  blekinge_crop_points.npz) + vasakronan blir riktiga referensfall
  med användarens facit som måttstock.
- Acceptans: kraven 1–4 ovan på BÅDA riktiga skanningarna + bänk-F1
  ≥ v4 överallt.

## Hållpunkter

- M1: carriers + arrangemang + cellklassning på vasakronan (lokalt).
- M2: full pipeline ingest→ifc, blekinge_crop under 60 s.
- M3: nytt GUI mot nya pipelinen.
- M4: serverkörning, användarvalidering, v5 blir default.
