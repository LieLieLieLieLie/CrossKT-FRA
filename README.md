# CrossKT-FRA

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.knosys.2024.112208-blue)](https://doi.org/10.1016/j.knosys.2024.112208)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **CrossKT-FRA**, the method presented in:

> Zilong Yin, Haoyu Wang, Bin Chen, Xin Zhang, Xiaogang Lin, Hangling Sun, Anji Li, and Chenyu Zhou. “Federated semi-supervised representation augmentation with cross-institutional knowledge transfer for healthcare collaboration.” *Knowledge-Based Systems*, 300:112208, 2024.

The peer-reviewed article and its publisher page link directly to this repository: [https://doi.org/10.1016/j.knosys.2024.112208](https://doi.org/10.1016/j.knosys.2024.112208).

## Overview

CrossKT-FRA is a vertical federated learning framework for healthcare collaboration with partially aligned samples. It combines:

1. federated representation learning over aligned samples;
2. supervised representation augmentation using federated knowledge and latent labels; and
3. indirect knowledge transfer from aligned data to local non-aligned data.

The repository includes implementations of CrossKT-FRA, VFedPCA, FedSVD, aggregation- and split-based VFL baselines, local baselines, downstream evaluation, and ablation experiments.

## Repository structure

```text
configs/          Command-line and data-split configuration
  args.py         Experiment arguments and default settings
  split.json      Dataset-specific split settings
dataset/          Dataset files used by the current experiments
model/            CrossKT-FRA components and baseline models
main.py           Main experiment entry point
performance.py    Evaluation, downstream, and ablation routines
utils.py          Data loading, splitting, and shared utilities
```

## Environment

The paper experiments used Python 3.9, PyTorch 2.0.0, CUDA 12.2, an NVIDIA RTX 4080 GPU, and 32 GB RAM. CUDA is optional; the code falls back to CPU when no compatible GPU is available.

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

## Data

The current loaders support tabular datasets configured in `configs/split.json`. The Wisconsin Diagnostic Breast Cancer data is included for convenience. MIMIC-III and any derived data must be accessed and used under the applicable PhysioNet credentialing, data-use, and citation requirements. Do not commit protected health information.

Other datasets referenced by the command-line choices are not bundled unless a corresponding directory is present. Place authorized data under `dataset/<dataset-name>/` and update `utils.py` if your local filenames differ.

## Running an experiment

Run commands from the repository root. For example:

```bash
python main.py --dataset MIMIC --split unequal --frl VFedPCA --gpu 0
```

A smaller public-data example is:

```bash
python main.py --dataset Breast --split equal --frl VFedPCA --gpu 0
```

Common options include:

- `--dataset`: dataset name defined in `configs/split.json`;
- `--split`: `equal` or `unequal`;
- `--frl`: `VFedPCA` or `FedSVD`;
- `--gpu`: CUDA device index (CPU fallback is automatic);
- `--experiment2`, `--experiment3`, and `--ablation`: experiment and ablation controls.

Outputs and learned checkpoints are written below `model/save/`. The active evaluation block in `main.py` determines which comparison or ablation routine is executed; enable only the experiment blocks you intend to reproduce.

## Reproducibility notes

- Dataset split sizes are defined in `configs/split.json`.
- Randomness is used in training and data splitting; report seeds and repeated-run statistics when comparing results.
- Dataset licenses and access terms are separate from the MIT license covering this repository's original source code.
- If you encounter a reproducibility problem, open a GitHub issue with the command, environment, dataset, and traceback.

## Citation

If this repository supports your work, please cite:

```bibtex
@article{yin2024crossktfra,
  title   = {Federated semi-supervised representation augmentation with cross-institutional knowledge transfer for healthcare collaboration},
  author  = {Yin, Zilong and Wang, Haoyu and Chen, Bin and Zhang, Xin and Lin, Xiaogang and Sun, Hangling and Li, Anji and Zhou, Chenyu},
  journal = {Knowledge-Based Systems},
  volume  = {300},
  pages   = {112208},
  year    = {2024},
  doi     = {10.1016/j.knosys.2024.112208}
}
```

## License

The original source code in this repository is released under the [MIT License](LICENSE). Third-party datasets and dependencies remain subject to their own licenses and terms.

## Maintenance

This repository is maintained as the public implementation associated with the published article. Constructive issues and reproducibility reports are welcome.
