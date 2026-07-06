# Actuarial NAM (`anam`)

An _inherently interpretable_ deep learning model for general insurance pricing. Each
covariate (and pairwise interaction) gets its own neural network, which get combined in a similar manner to the GAM, yet it learns nonlinear shape functions from the data instead of being told them.

$$
  \hat{y}_i = g^{-1}\bigl( \beta_0 + f_1(x_{i,1}) + f_2(x_{i,2}) + \dots + f_p(x_{i,p}) \bigr)
$$

`anam` fits these with a Poisson / Gamma / Gaussian likelihood, an exposure
offset (so frequency modelling is handled correctly, exactly as in a Poisson GLM),
optional _monotonicity_ constraints via monotonic lattice subnets, a roughness
penalty for smooth shape functions, and automatic selection of both main effects
and pairwise interactions.

This package is the reference implementation of the **Actuarial Neural Additive
Model** introduced in _An Interpretable Deep Learning Model for General Insurance
Pricing_ (Laub, Pho & Wong, 2026).

## Why an interpretable model

Deep learning has repeatedly outperformed GLMs on predictive accuracy for claim
modelling, but adoption in insurance is held back by the "black box" problem:
regulators, reinsurers and customers all need to see _why_ a premium is what it
is. `anam` is built on a concrete, mathematically formulated interpretability
framework (constrained optimisation under interpretability constraints, à la
Rudin) rather than post-hoc explanation. Because each $f_j$ is a dedicated
subnetwork, the model's full internal logic is exposed by construction: you can
read off every shape function and every interaction, weighted by where the data
actually is. The package imposes three practical interpretability requirements on
top of the NAM backbone:

- **Sparsity** — a three-stage variable-selection procedure (ensemble main-effect
  ranking, then interaction selection under the strong-heredity principle, then
  fine-tuning) trims the model to only the material terms.
- **Smoothness** — a roughness penalty (second-derivative, finite-difference
  approximated) keeps continuous shape functions from overfitting.
- **Monotonicity** — features that must move one way with the response (e.g. a
  bonus-malus score) are modelled with a constrained monotonic _lattice_ subnet,
  which guarantees the constraint at every point in the input domain — not merely
  penalised.

## Install

```bash
# with uv (recommended):
uv pip install anam @ git+ssh://git@github.com/Pat-Laub/anam.git
# or, with pip:
pip install "anam @ git+ssh://git@github.com/Pat-Laub/anam.git"
```

To hack on the source, clone and install editable:

```bash
git clone git@github.com:Pat-Laub/anam.git
cd anam && uv pip install -e .      # or: pip install -e .
```

Python ≥ 3.12, with `torch`, `numpy`, `pandas`, `matplotlib` and `scikit-learn`
(see [`pyproject.toml`](pyproject.toml)).

## Quickstart

`anam` works on the raw data: continuous vs categorical is read off the dtypes,
and standardising / encoding happens inside the model and is reused for
`predict` and plotting. Exposure enters as a multiplicative _offset_
($\log \mu = \beta_0 + \sum_j f_j(x_j) + \log(\text{expo})$), never as an
ordinary feature.

```python
from anam import ANAM

# Assuming X_train, y_train, X_val, y_val are loaded here.

model = ANAM(
    distribution="poisson",         # log link + Poisson NLL
    exposure="expo",                # offset column: log(expo) added to the predictor
    monotone={"bm": "increasing"},  # bonus-malus must be increasing
    smoothness=1e-3,                # roughness penalty on continuous main effects
)

# fit() runs variable selection by default (features="auto", interactions="auto")
# and then fine-tunes the chosen terms.
model.fit(X_train, y_train,
          X_val=X_val, y_val=y_val,
          epochs=300, batch_size=5000, lr=0.01, patience=40, random_state=2000)

model.features_, model.interactions_           # what survived selection
model.plot_selection()                         # both screening charts, elbow cutoffs marked
model.plot_effects(["ageph", "bm", "power"])   # shape functions, axes in original units
model.importance()                             # pd.Series, variance-based, sorted
model.predict(X_val)                            # E[claims], exposure offset included
```

### Assess, then select

The paper's workflow is to _look at the screening charts and then decide_, rather
than trust a fixed cutoff. Run the stages explicitly and pick the elbows yourself:

```python
model = ANAM(distribution="poisson", exposure="expo",
             monotone={"bm": "increasing"}, smoothness=1e-3)

model.rank_main_effects(X_train, y_train,
                         X_val=X_val, y_val=y_val, random_state=2000)
model.plot_main_effect_importance()           # variance-importance scree plot

model.rank_interactions(X_train, y_train,
                        X_val=X_val, y_val=y_val,
                        n_main=7, random_state=2000)
model.plot_interaction_importance()          # validation-loss improvement per pair

model.choose(n_main=7, n_interactions=6)      # lock in the cutoffs
model.fit(X_train, y_train,
          X_val=X_val, y_val=y_val,
          epochs=300, batch_size=5000, lr=0.01, patience=40, random_state=2000)
```

- `rank_main_effects` trains an ensemble of shallow NAMs and scores each
  covariate by the _variance of its shape function across the training data_ — a
  proxy for how much signal it carries — without yet committing to a cutoff.
- `rank_interactions` scores every pairwise term that passes the **strong
  heredity** rule (a pair is only considered if _both_ of its main effects
  survived the previous stage), measuring how much each candidate improves the
  validation loss over the mains-only baseline.
- `choose` freezes the cutoffs; `fit` then fine-tunes the chosen terms with the
  smoothness / monotonicity constraints active.

## Reading the results

Each shape curve is the feature's contribution to the linear predictor $g(\mu)$
(log-rate under a Poisson); on the original scale, $\exp(f_j)$ is a
_multiplier_ on the expected claim count.

```python
grid, values = model.effect("bm")          # raw curve (or category → value map)
model.plot_effect("bm")                     # single main effect
model.plot_effect("long", "lat")            # pairwise interaction surface
model.plot_effects(["ageph", "bm", "power"]) # panel of main effects
```

Categorical features give one value per level — drawn as a bar chart with the
real category labels, still in log-rate units — handled by an embedding rather
than an MLP, but read the same way: $\exp(f_\text{fuel})$ is a multiplier on the
expected count.

### Variable importance

The variance-based `importance()` measures the spread of each
shape function _across the actual training data_, so a curve that only swings
where there is no data scores near zero (≈ ×1.00 multiplier) and is correctly
flagged as negligible. Continuous-effect plots include a decile rug so the
well-supported region is visible at a glance.

## Development

```bash
uv run pytest tests/ -v
```

See `src/anam/` for the implementation. The
headline user-facing class `ANAM` lives in [`src/anam/api.py`](src/anam/api.py);
the model, lattice, losses, optimizer and sparsity modules underneath are the
computational core.

## Citation

If you use `anam` in your work, please cite the accompanying paper.

```bibtex
@article{laub2026anam,
  title   = {An interpretable deep learning model for general insurance pricing},
  author  = {Laub, Patrick J. and Pho, Tu and Wong, Bernard},
  journal = {Insurance: Mathematics and Economics},
  year    = {2026},
  pages   = {103270},
}
```