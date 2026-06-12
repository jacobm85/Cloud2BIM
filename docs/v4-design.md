# v4 — design från blankt blad

Mål (avgränsat med flit): från ett punktmoln detektera **väggar**,
**öppningar i väggar** och **bjälklag**. Inget annat. Pelare, trappor,
tak m.m. behålls i sina befintliga moduler.

## Varför inte bygga vidare på v1/v2/vertical/ML-vägen?

Fältfacit (2026-06-11): v1 är i särklass bäst på riktiga skanningar.
Syntetbänken säger samtidigt att ML-vägen är perfekt (F1=1.00) när
etiketterna är perfekta — och att v2 övergenererar och vertical hittar
noll. Slutsatsen är inte "v1:s matematik är bäst" utan:

1. **Etikettberoende är en svaghet, inte en styrka.** ML-vägen kastar
   alla punkter som inte är vägg-etiketterade. När modellen har fel
   (domängap mot S3DIS är stort i industri/vård/garage) försvinner
   väggevidensen helt. v1 använder alla punkter och påverkas inte.
2. **Ett enda tvärsnittsband är skört men ärligt.** v1 tittar på
   verklig punktnärvaro i höjdband. Det överlever skuggor bättre än
   linjeanpassning på glesa kluster.
3. **v1:s svaghet är känd**: möbler/klutter i bandet blir falska
   väggar, och fragmenteringen kräver aggressiv merge-logik.

v4 ska alltså förena v1:s etikettoberoende, helhöjdsevidens med ML som
*mjuk* prior och den globala regularisering som redan finns.

## Arkitektur: evidensraster ("konsensusplan")

### Bjälklag

1. Z-histogram med fin bin (2 cm) och **prominensbaserad** toppdetektering
   (`scipy.signal.find_peaks`) i stället för fast tröskel mot maxbinnen —
   en gles takyta i en industrihall överlever då bredvid ett massivt golv.
2. Varje topp förfinas med horisontell RANSAC (±bin) → ytnivå + stöd.
3. Ytor paras till fysiska bjälklag: tak-yta följd av golv-yta inom
   `max_slab_thickness` = ETT bjälklag (logik beprövad i slabs_ml).
4. ML-etiketter som **bonusvikt** på topp-poängen (×1+w·andel
   golv/tak-etiketter), aldrig som filter.
5. Nivåer som ger våningshöjd < min_storey_height: svagaste stödet ryker.

### Väggar — vertikal persistens i stället för ett band

1. XY-raster (3 cm). Storeyn delas i **N=12–16 Z-skivor**; per cell
   räknas hur många skivor som är ockuperade → *persistens* ∈ [0,1].
   - Vägg: ockuperad i nästan alla skivor (öppningar sänker till ~0.7).
   - Möbel/klutter: bara de nedersta skivorna → låg persistens.
   - Skugga: slumpvisa skivor bortfallna → persistensen sjunker lite,
     men kravet är relativt (andel av *ockuperade* skivor i området).
2. **Mjuk ML-viktning**: cellpoäng × (1 + α·väggandel − β·klutterandel),
   klippt till [0.5, 1.5]. 15 % etikettbrus flyttar poängen marginellt;
   en hård filtrering hade tappat 15 % av väggytan.
3. Tröskling → väggmask; morfologisk stängning broar skuggor.
4. Centrumlinjer ur masken med **RANSAC-linjepeeling** (samma metod som
   gav F1=1.00 i ML-vägen — den arbetar här på maskceller i stället för
   etikettfiltrerade punkter).
5. Tjocklek: perpendikulär utbredning av maskceller per segmentstation
   (median); enkelsidigt → singleton_thickness + utåtskjutning.
6. **Global regularisering** (befintlig `geometry/regularize.py`):
   riktningssnapp, kollineär merge, hörnslutning.

### Öppningar — frånvaro som evidens, etiketter som bekräftelse

1. Per vägg: fasadraster (along × z, 3 cm) av punkter inom
   tjockleksbandet runt väggplanet.
2. Kandidat = sammanhängande **tomt** rektangulärt område som inte rör
   rastrets kant (kantbortfall = skanningslucka, inte öppning).
3. Poäng = tomhet × geometrisk prior (dörr: når golv, 0.7–1.3 m bred,
   1.9–2.4 m hög; fönster: bröstning 0.5–1.2 m, rimliga mått) ×
   etikettbonus (dörr/fönster-etiketterade punkter i området höjer,
   krävs inte).
4. Klassning dörr/fönster på golvkontakt, inte på etikett.

## Beslutsgrunder / icke-mål

- Ingen träning, inga nya beroenden — numpy/scipy/cv2 + befintliga moduler.
- Global planlösningsoptimering (cellkomplex/rumsgraf) utvärderades men
  skjuts till v5: kräver rumsklassning som är svår att validera utan
  fler riktiga skanningar. v4:s regularisering ger 90 % av nyttan för
  10 % av risken.
- v4 exponeras som `algorithm="v4"` och blir default först när den slår
  v1 på den brusiga bänken OCH på minst en riktig skanning i wizarden.

## Acceptanskriterier (synthbench)

| Mätetal | Krav |
| --- | --- |
| Väggar F1, alla scenarier, `--noisy` | ≥ v1 och ≥ 0.90 |
| Öppningar R, `--noisy` | ≥ 0.75 (v1-nivå eller bättre) |
| Bjälklag nivåträff, kontor2v | 2/2 + rätt antal slabbar |
| Axelfel | ≤ 5 cm clean, ≤ 8 cm noisy |
