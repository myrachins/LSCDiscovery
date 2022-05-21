# GlossReader at LSCDiscovery: Train to Select a Proper Gloss in English -- Discover Lexical Semantic Change in Spanish

#### This is code for the GlossReader system in [LSCDiscovery](https://codalab.lisn.upsaclay.fr/competitions/2243) competition.
The 1st place in all subtasks except for the optional sense gain detection subtask.

## How to run

### Step 0: Prepare environment
1. Install [python](https://python.org/) 3.7 or later.
1. Clone repo and move to the root directory (i.e., `LSCDiscovery`).
1. Run `pip install -r requirements.txt` to install the required packages.
1. To run the code with the default parameters, download the precomputed vectors from [zenodo](https://zenodo.org/record/6569382).
1. Place all words files in `LSCDiscovery/data/words_vectors`.

### Step 1: Generate graded predictions
Run `python graded_scores.py` to generate our graded predictions (*GLM norm L1* from the results table).
Additionally, you can change the default parameters: 
- `--words-vectors-dir` - computed vectors for usages of the target words.
- `--output-path` - path where to store predictions.
- `--normalize` - normalize usages' vectors.
- `--norm_ord` - order of the norm.

### Step 2: Generate binary predictions
Run `python binary_scores.py` to generate our binary predictions (*Thres. all* from the results table).
Additionally, you can change the default parameters: 
- `--graded-scores-path` - graded change scores from the previous step.
- `--output-path` - path where to store predictions.
- `--threshold` - threshold for binarization.
  - *Thres. all*: 0.7015 (default)
  - *Thres. revealed*: 0.717
  - *Thres. all, fixed*: 0.667

### Step 3: Evaluate graded predictions
Run `python evaluate_graded.py` to calculate graded test scores for a prediction file.
Additionally, you can change the default parameters: 
- `--preds-graded-path` - path to the generated graded predictions.
- `--gold-scores-path` - path to the gold scores. 
- `--target-words-path` - path to the list of target words used in the evaluation. 

### Step 4: Evaluate binary predictions
Run `python evaluate_binary.py` to calculate binary test scores for a prediction file.
Additionally, you can change the default parameters: 
- `--preds-binary-path` - path to the generated binary predictions.
- `--gold-scores-path` - path to the gold scores. 
- `--target-words-path` - path to the list of target words used in the evaluation.

## Authors
- Maxim Rachinskiy
- Nikolay Arefyev
