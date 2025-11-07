# PyGEDAI Usage Guide

This library implements the Generalized Eigenvalue De-Artifacting Instrument (GEDAI) for EEG cleaning. The core API mirrors the original MATLAB tooling while embracing PyTorch tensors for efficient numerical work. This document provides a concise reference for integrating `gedai()` and `batch_gedai()` into your projects while staying faithful to the algorithmic description in Ros et al. (2025).

---

## What GEDAI Does

GEDAI (Generalized Eigenvalue De-Artifacting Instrument) is an unsupervised, theoretically grounded denoiser for heavily contaminated EEG. It contrasts each epoch’s covariance with a physics-based forward model (leadfield), retaining only components that behave like genuine neural activity—no clean calibration data or manual supervision required.

**Core mechanism:** A generalized eigenvalue decomposition (GEVD) compares the data covariance (`dataCOV`) with a leadfield-derived reference covariance (`refCOV`). Components aligned with the brain subspace are kept; orthogonal components are treated as artifacts. The Signal & Noise Subspace Alignment Index (SENSAI) automatically selects the optimal rejection threshold. See [Ros et al., 2025](https://doi.org/10.1101/2025.10.04.680449) for full details.

---

## Background and References

- Ros, T., Férat, V., Huang, Y., Colangelo, C., Kia, S. M., Wolfers, T., Vulliemoz, S., & Michela, A. (2025). *Return of the GEDAI: Unsupervised EEG Denoising based on Leadfield Filtering*. bioRxiv. https://doi.org/10.1101/2025.10.04.680449
- Original MATLAB/EEGLAB plugin: https://github.com/neurotuning/GEDAI-master (this Python port follows the architecture and processing stages documented there).


---

## Quick Start

```python
import torch
from GEDAI import gedai, batch_gedai

eeg = torch.load("test.pt") # (channels, samples)
leadfield = torch.load("leadfield.pt") # (channels, channels)

# Clean a single recording
result = gedai(eeg, sfreq=100.0, leadfield=leadfield)
cleaned = result["cleaned"]

# Clean a mini-batch of trials
batch = eeg.unsqueeze(0) # add batch dimension
cleaned_batch = batch_gedai(batch, sfreq=100.0, leadfield=leadfield)

# result contains cleaned EEG, per-band thresholds, and SENSAI quality metrics
# Typical SENSAI scores range from ~0.3 to 0.8 on cleanable data
print(f"SENSAI quality score: {result['sensai_score']:.3f}")
```

The notebook `testing/HBN.ipynb` covers an end-to-end example, including plots that compare raw and cleaned signals.

---

## `gedai()`

`gedai(eeg, sfreq, denoising_strength="auto", epoch_size=1.0, leadfield=None, *, wavelet_levels=9, matlab_levels=None, chanlabels=None, device="cpu", dtype=torch.float32, skip_checks_and_return_cleaned_only=False, batched=False, verbose_timing=False, TolX=1e-1, maxiter=500)`

### Purpose

Execute the GEDAI pipeline on a single EEG recording shaped `(channels, samples)` by applying rank-safe referencing, broadband denoising, multi-resolution wavelet cleaning, and artifact scoring.

### Required Parameters

- `eeg`: PyTorch tensor or array-like with shape `(channels, samples)`. For best performance convert to a torch tensor before calling.
- `sfreq`: Sampling frequency in Hertz. Guides epoch sizing and band selection.
- `leadfield`: `(channels, channels)` reference covariance matrix derived from your EEG forward model (leadfield) that defines the theoretical brain signal subspace. Accepts a filepath, numpy array, or torch tensor. The row and column order must match the EEG channel ordering because tensors carry no channel labels.

### Key Optional Parameters

- `denoising_strength`: Controls artifact suppression aggressiveness.
  - `"auto"` (default): SENSAI-optimized threshold (noise multiplier = 3.0).
  - `"auto-"`: More aggressive filtering (noise multiplier = 6.0).
  - `"auto+"`: More conservative filtering (noise multiplier = 1.0).
  - Numeric value (`0.0–12.0` typical): Manual threshold passed directly to the optimizer.
  Internally this value is forwarded to `artifact_threshold_type` in `gedai_per_band()`.
- `epoch_size`: Desired epoch duration in seconds. The helper enforces an even number of samples via `_ensure_even_epoch_size`, padding and trimming as necessary.
- `wavelet_levels`: Number of Haar MODWT levels when `matlab_levels` is `None`. Typical values fall between `7` and `9`.
- `matlab_levels`: Alternative to `wavelet_levels`, recreating MATLAB level numbering with `2**matlab_levels + 1` bands. Leave `None` unless porting MATLAB scripts directly.
- `chanlabels`: Placeholder for channel label remapping. Currently not implemented and raises an error when supplied.
- `device`: Target torch device such as `"cpu"` or `"cuda"`. EEG data, leadfield, and internal buffers move to this device.
- `dtype`: Torch dtype used during computation, defaulting to `torch.float32` for a balanced memory and compute footprint; set `torch.float64` when maximum numerical accuracy is required and resources permit.
- `skip_checks_and_return_cleaned_only`: When `True`, bypass validation and return only the cleaned tensor to reduce overhead.
- `batched`: Internal flag used by `batch_gedai()`. Leave `False` in user-facing calls.
- `verbose_timing`: Enables profiling markers emitted by `profiling.py`, useful for benchmarking.
- `TolX`: Convergence tolerance for the golden-section search used during automatic thresholding (default `1e-1`).
- `maxiter`: Maximum iterations allowed for the threshold optimizer (default `500`).

### Returns

By default returns a dictionary with:

- `cleaned`: Denoised EEG `(channels, samples)`.
- `artifacts`: Removed components (`input_referenced - cleaned`).
- `sensai_score`: Overall quality metric (higher means better alignment with the brain subspace).
- `sensai_score_per_band`: Per-band SENSAI scores (length = number of wavelet bands plus the broadband pass).
- `artifact_threshold_per_band`: Thresholds applied to each wavelet band.
- `artifact_threshold_broadband`: Threshold used during the initial broadband pass.
- `epoch_size_used`: Actual epoch duration in seconds after enforcing an even sample count.
- `refCOV`: Reference covariance matrix used for GEVD.

When `skip_checks_and_return_cleaned_only=True`, the function returns only the `cleaned` tensor.

### Typical Workflow

1. Load or calculate a `(channels, channels)` leadfield covariance from `leadfield_calibrated/` or your own pipeline.
2. Convert raw EEG to a torch tensor and send it to the intended device and dtype.
3. Call `gedai(...)` and capture the cleaned signal along with diagnostics.
4. Visualize the results using utilities such as `plot_eeg` in `testing/HBN.ipynb`.

---

## `batch_gedai()`

`batch_gedai(eeg_batch, sfreq, denoising_strength="auto", epoch_size=1.0, leadfield=None, *, wavelet_levels=9, matlab_levels=None, chanlabels=None, device="cpu", dtype=torch.float32, parallel=True, max_workers=None, verbose_timing=False, TolX=1e-1, maxiter=500)`

### Purpose

Vectorize the GEDAI pipeline across a batch dimension. Input tensors must be shaped `(batch, channels, samples)`. Each sample is processed independently, optionally in parallel via a thread pool.

### Required Parameters

- `eeg_batch`: PyTorch tensor containing EEG recordings arranged as `(batch, channels, samples)`.
- `sfreq`: Sampling frequency shared across the batch.
- `leadfield`: `(channels, channels)` reference covariance matrix reused for every batch element. Ensure its row and column order mirrors the channel order in `eeg_batch`.

### Key Optional Parameters

- `denoising_strength`, `epoch_size`, `wavelet_levels`, `matlab_levels`, `chanlabels`, `device`, `dtype`, `TolX`, `maxiter`: Match the semantics of the corresponding arguments on `gedai()` and are forwarded per sample.
- `parallel`: When `True`, executes each batch element in a `ThreadPoolExecutor`. Set to `False` for serial execution or debugging.
- `max_workers`: Overrides the number of worker threads when `parallel` is enabled. Defaults to Python's heuristic based on CPU count.
- `verbose_timing`: Aggregates profiling information across the batch to assist throughput measurements.

### Returns

PyTorch tensor shaped `(batch, channels, samples)` containing the cleaned EEG for each input sample. Internally the function gathers the `cleaned` value from each `gedai()` call and stacks the results.

### Usage Example

```python
from pathlib import Path
import torch

project_root = Path.cwd()
eeg_trial = torch.load(project_root / "testing" / "test.pt")
leadfield = torch.load(project_root / "leadfield_calibrated" / "refCov_GSN_HydroCel_129.pt")

batch = eeg_trial.unsqueeze(0)
cleaned = batch_gedai(batch, sfreq=100.0, leadfield=leadfield, verbose_timing=True)
```

This mirrors the workflow shown in `testing/HBN.ipynb`, where the cleaned batch is plotted against the raw recording.

---

## Tips and Troubleshooting

- Ensure the leadfield reference covariance shape matches the EEG channel count. The functions raise a `ValueError` when dimensions disagree.
- **Channel order is critical:** Double-check that the row and column order of the reference covariance matches your EEG channel order (e.g., channel index 0 in both tensors corresponds to C1); misalignment silently degrades cleaning quality.
- The `leadfield` parameter should be a reference covariance computed from a forward model of your montage. Precomputed examples are available in `leadfield_calibrated/`.
- **Average referencing:** GEDAI applies a non-rank-deficient average reference (division by `n_channels + 1`) to avoid ICA ghost components—do not average-reference your data beforehand. See [Kim et al., 2023](https://doi.org/10.3389/frsip.2023.1064138) for background.
- The default threshold search uses golden-section (`"parabolic"`). An optional debug mode (`"grid"`) exhaustively evaluates thresholds from 0.0 to 12.0 in 0.1 steps and is roughly 100× slower.
- GEDAI typically processes ~1 s of 64-channel EEG in 0.5–2 s on CPU, depending on `wavelet_levels` and `denoising_strength`. GPU acceleration provides limited benefit because the matrices are modest in size.
- Use `batch_gedai()` for multiple independent trials or subjects; with `parallel=True` and adequate CPU cores, throughput scales nearly linearly with batch size.
- If thread-pool contention or hangs arise when running `batch_gedai()` in parallel mode, set single-threaded math libraries before importing torch:
  ```python
  import os
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
  os.environ["BLIS_NUM_THREADS"] = "1"

  import torch
  try:
      torch.set_num_threads(1)          # intra-op
      torch.set_num_interop_threads(1)  # inter-op
  except RuntimeError:
      pass  # torch was already initialised
  ```
  The `set_num_threads` calls must run before PyTorch initialises; future Python 3.14+ releases are expected to reduce the need for this workaround.
- The pipeline enforces even epoch lengths. If the requested epoch and sampling rate yield an odd sample count, GEDAI pads before processing and trims afterward.
- When running on GPU, move both EEG data and leadfield tensors to the target device prior to calling the API.
- Enable `verbose_timing=True` during development to gather profiling markers such as `start_batch`, `modwt_analysis`, and `batch_done`.
- If you only require cleaned signals, set `skip_checks_and_return_cleaned_only=True` to avoid collecting diagnostic metadata.

---

## Citation

Ros, T., Férat, V., Huang, Y., Colangelo, C., Kia, S. M., Wolfers, T., Vulliemoz, S., & Michela, A. (2025). *Return of the GEDAI: Unsupervised EEG Denoising based on Leadfield Filtering*. bioRxiv. https://doi.org/10.1101/2025.10.04.680449

When referencing this Python package, please also acknowledge that it ports the original MATLAB/EEGLAB plugin available at https://github.com/neurotuning/GEDAI-master.

---

## License

This port follows the PolyForm Noncommercial License 1.0.0, identical to the original GEDAI plugin. The core algorithms are patent pending; commercial use requires obtaining the appropriate license from the patent holders. See `LICENSE` for full terms and contact information.

---

## Further Resources

- `GEDAI.py`: Core implementation with inline comments describing the Haar MODWT pipeline, SEnSAI scoring, and artifact reconstruction.
- `auxiliaries/`: Helper modules including `GEDAI_per_band.py`, `SENSAI_basic.py`, and `clean_EEG.py`, which provide per-band denoisers and optimization routines.
- `testing/HBN.ipynb`: Practical notebook demonstrating data loading, covariance handling, calls to `batch_gedai()`, and visualization of results.

For issues or feature requests, please open a GitHub issue in this repository.
