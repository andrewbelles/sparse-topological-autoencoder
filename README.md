# Topological Sparse Representations for FMA Music

This repository studies whether topology-preserving sparse representations of a learned music manifold retain genre-relevant structure. The current project direction is:

1. preprocess FMA-small audio into log-normalized mel-spectrogram tensors,
2. learn a fixed Audio Barlow Twins / SupCon contrastive anchor manifold,
3. sparsify that manifold with Top-s sparse autoencoders,
4. compare ordinary sparse codes against topology-regularized sparse codes, and
5. evaluate whether topology preservation improves linear, local, and PH-feature probes.

The main scientific question is not simply whether a model classifies genre well. It is whether preserving persistent-homology structure under sparsification gives a useful and defensible signal about genre structure in the learned ABT manifold.

## Repository Map

- `preprocess/`: FMA metadata/audio download helpers and log-mel tensor generation.
- `representation/`: Audio Barlow Twins training, embedding extraction, manifold selection, and UMAP visualization.
- `compression/`: projection baselines plus the current sparse-dictionary SAE/topo-SAE models.
- `persistence/`: persistence diagrams, residual diagrams, and within/between-genre topology variation.
- `evaluation/`: linear probes, topology preservation scoring, sparse-dictionary diagnostics, PH-transfer probes, and coordinate+PH hybrid probes.
- `configs/`: YAML configs for each stage. Runtime artifacts under `data/`, `images/`, and checkpoints are ignored.

## Data And Preprocessing

The expected dataset is FMA-small: 8 top-level genres with 1,000 tracks per genre. The preprocessing path downsamples audio to `22.05 kHz`, computes `64`-bin mel-spectrograms, applies log scaling, then min-max normalizes each track.

Download metadata:

```bash
bash -v preprocess/meta.sh
```

Download and extract FMA-small:

```bash
bash -v preprocess/small.sh
```

Generate mel tensors:

```bash
python -m preprocess.mel -d preprocess/data/fma_small
```

Add `--sample-images` to write a few preview spectrograms to `preprocess/images/`.

## Anchor Manifold

The anchor manifold is produced by `representation.manifold`. It trains the configured Audio Barlow Twins grid, optionally adds SupCon regularization, extracts track-level embeddings, evaluates candidate manifolds on validation probes, and writes the selected manifold as:

```text
representation/data/anchor_fma_small_mel_training.parquet
representation/data/anchor_fma_small_mel_validation.parquet
representation/data/anchor_fma_small_mel_test.parquet
```

Run:

```bash
python -m representation.manifold -d preprocess/data/fma_small_mel
```

Plot the selected anchor manifold:

```bash
python -m representation.umap -a representation/data
```

The selected `anchor` is the fixed reference space for the current experiments. Downstream sparse dictionaries and topology metrics should be interpreted relative to this anchor, not raw mel coordinates.

## Sparse Dictionary Models

The active compression direction is sparse dictionary learning over ABT anchor embeddings. `compression.sparse_dictionary` trains two Top-s sparse autoencoder variants:

- `sae`: reconstruction + SupCon on sparse codes.
- `topo_sae`: reconstruction + SupCon + topology regularization on sparse codes.

The SAE architecture is a linear Top-s dictionary model:

```text
h = W_e (z - b_0) + b_e
a = TopS(ReLU(h), s)
z_hat = W_d a + b_0
```

Decoder atoms are normalized so sparsity is meaningful. The current topo-SAE topology penalty uses differentiable persistence-image distance over `H0/H1`, with ABT-reference median scaling and a small scale penalty to avoid misleading gradients from sparse-code radius mismatch.

Run the sparse dictionary sweep:

```bash
python -m compression.sparse_dictionary -a representation/data -c configs/sparse_dictionary.yaml
```

Outputs are single multi-run parquets per method, for example:

```text
compression/data/sae_anchor_fma_small_mel.parquet
compression/data/topo_sae_anchor_fma_small_mel.parquet
```

## Persistence Analysis

Compute per-genre persistence diagrams and residual diagrams for the selected anchor:

```bash
python -m persistence.diagrams --mel-dir preprocess/data/fma_small_mel
```

Measure topology variation within and across genres:

```bash
python -m persistence.variation -a representation/data
```

These scripts write summaries to `persistence/data/` and plots to `persistence/images/`. The variation analysis is useful for checking whether topology is meaningfully genre-separated before asking sparse models to preserve it.

## Evaluation

Linear probe evaluation is the primary downstream metric. For sparse dictionaries, use the sparse-dictionary linear config:

```bash
python -m evaluation.linear -d compression/data -c configs/linear_sparse_dictionary.yaml
```

Sparse dictionary diagnostics compute reconstruction error, active coefficient counts, persistence-image/Betti/Wasserstein topology distances, and support-frequency visualizations:

```bash
python -m evaluation.sparse_dictionary -d compression/data -c configs/sparse_dictionary_eval.yaml
```

PH-transfer probes ask whether local persistent-homology features alone carry genre information:

```bash
python -m evaluation.transfer -d compression/data -c configs/transfer.yaml
```

Hybrid probes concatenate coordinate features and local PH features with block standardization:

```bash
python -m evaluation.hybrid -d compression/data -c configs/hybrid.yaml
```

The main comparison table to inspect is typically:

```text
evaluation/data/sparse_dictionary_compact_summary.csv
```

Important metrics:

- `f1`, `pr_auc`: downstream genre probe performance.
- `pi_Z_A`: persistence-image distance between ABT anchor `Z` and sparse codes `A`.
- `pi_Z_Zhat`: persistence-image distance between ABT anchor `Z` and reconstructed anchor coordinates `Z_hat`.
- `recon_mse`: reconstruction error in anchor space.
- `avg_active`, `med_active`: effective sparse support size.

Lower topology distance is better; higher `f1` and `pr_auc` are better.

## Optional Projection Baselines

Classical and manifold projection baselines remain available for comparison, but they are no longer the central experiment.

Generate fixed CS projections:

```bash
python -m compression.project -a representation/data
```

Evaluate projections:

```bash
python -m evaluation.linear -d compression/data -c configs/linear.yaml
python -m evaluation.topology -d compression/data
```

Run a configured scheme end-to-end:

```bash
python -m evaluation.scheme -c configs/scheme.yaml
```

## Citation

This dataset was made possible by the work of Defferrard et al. If you use FMA downstream, credit the original dataset authors:

> Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson.  
> **"FMA: A Dataset for Music Analysis"**  
> *18th International Society for Music Information Retrieval Conference (ISMIR), 2017.*  
> [Official FMA GitHub Repository](https://github.com/mdeff/fma)
