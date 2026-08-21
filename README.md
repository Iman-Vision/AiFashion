# AI Fashion Accessorizer

Upload a clothing photo (or a few), confirm what it actually is, and get
mood boards pairing it with real matched pieces from a fashion dataset —
plus an opt-in sidebar for accessories and an outwear layer.

## How it works

### 1. Detection — "what did you upload?" (then you confirm it)

`ai_fashion/detector.py` guesses each uploaded photo's category — **top,
outwear, bottom, shoes** (no "dress" category — dress-like uploads default
to "top"). Detection tries three things in order, falling back gracefully:

1. **`models/clothing_classifier.pt`** — a MobileNetV3-Small CNN, fine-tuned
   on however many photos are available per category (see
   [Rebuilding the data](#rebuilding-the-data-only-needed-if-you-want-to-change-it)
   below). Scores well on catalog-style photos, but the training set is
   entirely *flat product-catalog shots* (garment centered, filling the
   frame, white background) — real lifestyle photos (a person wearing the
   item, off-center, cluttered background) are out-of-distribution and can
   get confidently misclassified (a real photo of a person in a blouse was
   once scored "shoes" at 99.97% confidence). A heavier-augmentation
   training variant was tried to close that gap and reverted — it lowered
   confidence without fixing the actual misclassifications, so it wasn't
   worth the tradeoff. **The real fix here isn't a training flag: the app
   always shows a confirm step** (`templates/confirm.html`, routes
   `/analyze` → `/generate` in `app.py`) — every uploaded photo displays
   its guessed category with a confidence percentage and lets you correct
   it (or, for a single upload, mark it "Mixed / full outfit" to use
   purely as a style/color reference without locking it into any one
   slot) before any board gets built. Wrong guesses now cost one click
   instead of silently producing a wrong board.
2. **MobileNet + ImageNet mapping** — if the classifier checkpoint is
   missing or incompatible (e.g. its label count doesn't match
   `CATEGORY_LABELS` after a code change), falls back to a generic
   ImageNet-pretrained MobileNet, with ~16 of its 1000 classes crudely
   mapped onto our 4 categories. Much weaker, but keeps the app working.
3. **Filename/aspect-ratio heuristics** — last resort if even that fails.

### 2. Feature extraction — "what does it look like?"

`ai_fashion/analyzer.py` (`analyze_image`) turns a photo into a handcrafted
feature dict, no ML model required:

- **Dominant color** — most frequent color in the image, *excluding* the
  white/grey studio backdrop almost every product photo is shot on (see
  `_is_studio_background`). Without that exclusion, nearly every item's
  "color" comes out white regardless of the actual garment.
- **Pattern** — solid / checked / striped / print, from row/column
  brightness-transition frequency.
- **Texture** — smooth / textured, from edge intensity.

These get turned into a numeric vector by `feature_vector()`:
`[R, G, B, pattern one-hot ×5, texture one-hot ×2, formality one-hot ×2]`,
with color weighted heaviest, formality second, pattern/texture as
tie-breakers only (see the weight comments in `analyzer.py` for why —
getting this balance wrong is what made matching feel "random" earlier).

**Formality** (casual vs. dressy) is what stops sneakers-with-a-skirt or
heels-with-jeans:
- Bottoms: **pants → casual, skirt → dressy** (real data — Re-PolyVore kept
  them in separate folders).
- Shoes: no such folder split exists, so `estimate_shoe_formality()` uses a
  silhouette heuristic — tall/narrow (heel-shaped) → dressy, wide/flat-soled
  → casual. Approximate, not perfect.
- Tops/outwear: formality-neutral (doesn't pull matching either way).

### 3. Matching — "what goes with it?"

`ai_fashion/recommender.py` loads `models/gallery_index.json` — ~6,700 real
dataset photos (tops, outwear, bottoms, shoes, accessories), each
pre-vectorized with the same `feature_vector()` function above — into
`ITEM_POOL`.

For a detected item, `build_boards()`:
1. Figures out which of the three **essential slots** (top / bottom / shoes)
   are still missing.
2. Pulls the top ~9 pool candidates per missing slot by cosine similarity to
   the uploaded item.
3. Scores every *combination* of candidates — not just each one against the
   uploaded item independently, but against **each other too**
   (`_combo_score`) — so the bottom and shoes it picks actually agree with
   one another (same formality, complementary color), not just each
   coincidentally resembling the top.
4. Returns the top 3 non-overlapping combos as three mood boards.

**Outwear is optional, never forced.** It only appears on a board if you
actually uploaded one — it has its own slot (`UPLOADED_SLOT`), separate from
"top", and is deliberately left out of `ESSENTIAL_SLOTS` so it's never
auto-filled from the pool. The sidebar (below) is the only way to add a
*pool* outwear piece to an existing board, and only if it actually fits.

### 4. Multi-photo mode

Upload 2–3 of your own photos (e.g. your actual top + pants + shoes) and
`recommend_from_uploads()` builds **one board from your real pieces**,
filling in only whatever essential slot you didn't provide.

### 5. Sidebar — accessories & outwear, opt-in only, per-board

Accessories and an extra outwear layer are **never baked into the initial
board**. The sidebar lists every accessory category *as it actually exists
in the dataset* (bag, bracelet, brooch, earrings, eyewear, gloves, hairwear,
hats, necklace, neckwear, rings, watches) plus an outwear-layer checkbox —
no vague umbrella groupings.

Tick whichever you want and hit "Add Selected". For **each of the 3
boards independently**, and for each ticked category, it calls `/suggest`
with that specific board's own pieces as context (`suggest_additions()`) —
so a bracelet that suits board 1's color palette but clashes with board 2's
isn't forced onto both. Each candidate carries a `fits` flag (cosine
similarity ≥ `FIT_THRESHOLD`, currently `0.55`, against that board's own
pieces):

- **Outwear that fits** → splices a real piece card directly into that
  board's grid.
- **Anything that fits (accessories)** → appended as a small chip in that
  board's footer strip.
- **Doesn't fit** → skipped for that board, and named in the "skipped"
  note under the button (e.g. "Board 2: skipped Rings, Hats") — never
  forced on.

The board grid itself uses CSS Grid (`.piece-grid`, `auto-fit` columns),
not hand-placed pixel coordinates — pieces keep a slight rotation each for
the scattered-photos look, but the grid guarantees cards can never overlap,
regardless of container width or how many pieces/add-ons a board ends up
with. (An earlier pixel-coordinate version did overlap under real layout
conditions — that's why this changed.)

## Running it

### Docker (recommended)
```
docker compose up -d
```
Serves at `http://localhost:5000`. The whole project directory is
bind-mounted into the container, so code edits apply on `docker compose
restart` — no rebuild needed unless `requirements.txt` changes.

### Local venv
```
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```
Or just run `run.bat` on Windows — it does all three steps.

### CLI (no web UI)
```
python main.py --image path/to/photo.jpg
```
Writes standalone HTML mood boards to `out/`.

## Rebuilding the data (only needed if you want to change it)

The original 3.5GB Re-PolyVore dataset dump has been **deleted** — only the
small subset actually used got copied into `gallery_assets/` (~200MB). Both
scripts below read from `gallery_assets/`, since that's all that's left —
**this caps classifier training data at whatever was originally sampled
into `gallery_assets/` per category (currently ~400), not the 4,000/category
the flags below ask for; they silently take everything available instead.**
Re-downloading the full Re-PolyVore dump is the only way to get more.

**`build_gallery_index.py`** — rebuilds `models/gallery_index.json` (the
matching pool), re-extracts color/pattern/texture/formality for every item,
and produces a background-removed `*_cutout.png` cutout of it (flood-fill
from the four corners with a color-distance threshold — see
`analyzer.remove_background`) for display on the moodboard; the original
JPG is left alone alongside it (`build_classifier_dataset.py` still needs
it, and background doesn't matter for classification). Works well for the
solid-color studio backdrops most of this dataset is shot on, including
non-white ones (a shirt on a black backdrop was showing as a solid black
rectangle before this); does nothing useful for busy/on-model photos, which
just keep their original background. Takes a few minutes for the full pool
(~130ms/image):
```
python build_gallery_index.py --per-category 400
```
Uploaded photos get the same cutout treatment on the fly at request time
(`app.py`'s `_cutout_url`), so uploaded pieces look visually consistent
with the matched pool items on the board.

**`build_classifier_dataset.py`** + **`train_all.py`** — rebuilds the
classifier training set and retrains it:
```
python build_classifier_dataset.py --per-category 4000
python train_all.py --data-dir dataset --epochs-classifier 15
```

If you want to start over with the *full* original dataset (more training
variety, real pants-vs-skirt-vs-heels labels beyond what's already
captured), you'd need to re-download/extract Re-PolyVore and point `--root`
at it again for **both** scripts — see their `DEFAULT_ROOT`.

## What `models/clothing_classifier.pt` actually is

It's *only* the item-type detector's weights — the CNN that answers "is
this a top, outwear, bottom, or shoes?" (step 1 above), and only the first
opinion in a chain the user gets to correct via the confirm step — not an
oracle. It has nothing to do with matching/pairing; that's pure
cosine-similarity math over handcrafted feature vectors (step 3), no
learned model involved. If this file is missing or was trained with a
different category list, detection silently degrades to the weaker
MobileNet/heuristic fallback — the app still works, just less accurately
(and the confirm step catches it either way).

## Project layout

```
ai_fashion/
  analyzer.py      color/pattern/texture/formality extraction, feature vectors
  detector.py       item-type detection (classifier -> MobileNet -> heuristic)
  classifier.py      the MobileNetV3 classifier: model, training, inference
  recommender.py     matching/pairing logic, ITEM_POOL, board building
  moodboard.py        standalone HTML board renderer (used by main.py only)
app.py                 Flask web app (routes: /, /analyze, /generate, /suggest, /gallery, /uploads)
main.py                 CLI entry point
build_gallery_index.py     builds models/gallery_index.json + gallery_assets/
build_classifier_dataset.py builds dataset/ for classifier training
train_all.py                 runs classifier training
templates/index.html         upload page
templates/confirm.html        confirm/correct detected type before generating
templates/board.html           mood boards + sidebar
static/                CSS/JS
models/                classifier.pt + gallery_index.json (committed-size, not gitignored)
gallery_assets/        matched-item photos served at /gallery/... (gitignored, ~200MB)
dataset/                classifier training images (gitignored)
```

## Known limitations

- **Real-world photo accuracy**: the classifier is trained entirely on flat
  product-catalog shots (see step 1 above) — it can still misclassify a
  real "person wearing the item" photo, even confidently (augmentation was
  tried and reverted — it reduced confidence without fixing the
  misclassifications). This is why the app always shows a confirm/correct
  step rather than trusting a single guess.
- Shoe formality is a silhouette *heuristic*, not ground truth — it will
  misjudge some shoes (e.g. a flat but narrow dress shoe, or a chunky
  platform heel).
- Background removal (`remove_background`) is a corner-flood-fill
  heuristic, not real segmentation — it only removes regions connected to
  a corner within a color-distance threshold. A solid studio backdrop
  (white, black, grey — most of the dataset) comes out clean; a busy or
  on-model photo just keeps its original background since there's no
  matching corner region to flood from.
- The classifier only distinguishes top/outwear/bottom/shoes — pants vs.
  skirt within "bottom" isn't detected from an uploaded photo, only from
  the pool (where the source folder is known).
- `gallery_assets/` and `dataset/` are both gitignored — cloning this repo
  fresh needs `build_gallery_index.py` / `build_classifier_dataset.py` run
  again against a re-downloaded Re-PolyVore dump, since the original was
  deleted.
