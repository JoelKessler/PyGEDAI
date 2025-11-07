# PyGEDAI Usage Guide

This library implements the Generalized Eigenvalue De-Artifacting Instrument (GEDAI) for EEG cleaning. The core API mirrors the original MATLAB tooling while embracing PyTorch tensors for efficient numerical work. This document provides a concise reference for integrating `gedai()` and `batch_gedai()` into your projects while staying faithful to the algorithmic description in Ros et al. (2025).

---

## What GEDAI Does

GEDAI (Generalized Eigenvalue De-Artifacting Instrument) is an unsupervised, theoretically-grounded algorithm that cleans highly contaminated EEG data by automatically separating brain signals from artifacts—without requiring clean reference data or expert supervision.

---

## Background and References

- Ros, T, Férat, V., Huang, Y., Colangelo, C., Kia S.M., Wolfers T., Vulliemoz, S., & Michela, A. (2025). *Return of the GEDAI: Unsupervised EEG Denoising based on Leadfield Filtering*. bioRxiv. https://doi.org/10.1101/2025.10.04.680449
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
- `leadfield`: `(channels, channels)` tensor, numpy array, or filepath pointing to a serialized covariance matrix that drives artifact estimation. The row and column order must match the EEG channel ordering because tensors carry no channel labels.

### Key Optional Parameters

- `denoising_strength`: Controls broadband artifact suppression. Accepts `"auto"`, `"auto-"`, or MATLAB-style numeric and qualitative presets.
- `epoch_size`: Desired epoch duration in seconds. The helper enforces an even number of samples via `_ensure_even_epoch_size`, padding and trimming as necessary.
- `wavelet_levels`: Number of Haar MODWT levels when `matlab_levels` is `None`. Typical values fall between `7` and `9`.
- `matlab_levels`: Alternative to `wavelet_levels`, recreating MATLAB level numbering with `2**matlab_levels + 1` bands. Leave `None` unless porting MATLAB scripts directly.
- `chanlabels`: Placeholder for channel label remapping. Currently not implemented and raises an error when supplied.
- `device`: Target torch device such as `"cpu"` or `"cuda"`. EEG data, leadfield, and internal buffers move to this device.
- `dtype`: Torch dtype used during computation, defaulting to `torch.float32` for a balanced memory and compute footprint; set `torch.float64` when maximum numerical accuracy is required and resources permit.
- `skip_checks_and_return_cleaned_only`: When `True`, bypass validation and return only the cleaned tensor to reduce overhead.
- `batched`: Internal flag used by `batch_gedai()`. Leave `False` in user-facing calls.
- `verbose_timing`: Enables profiling markers emitted by `profiling.py`, useful for benchmarking.
- `TolX`: Convergence tolerance forwarded to the per-band optimizer defined in `SENSAI_basic`.
- `maxiter`: Maximum iterations permitted for the per-band optimization routine.

### Returns

By default the function returns a dictionary with:

- `cleaned`: Denoised EEG tensor shaped `(channels, samples)`.
- `artifacts`: Residual artifacts computed as `referenced input - cleaned`.
- `sensai_score`: Aggregate SENSory Artifact Index value.
- `sensai_score_per_band`: Tensor of SEnsAI scores per wavelet band.
- `artifact_threshold_per_band`: Tensor containing per-band artifact thresholds.
- `artifact_threshold_broadband`: Scalar threshold used during the initial broadband denoising pass.
- `epoch_size_used`: Final epoch duration in seconds after enforcing an even sample count.
- `refCOV`: Reference covariance tensor derived from the provided leadfield.

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
- `leadfield`: `(channels, channels)` reference covariance tensor reused for every batch element. Ensure its row and column order mirrors the channel order in `eeg_batch`.

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

- Ensure the leadfield covariance shape matches the EEG channel count. The functions raise a `ValueError` when dimensions disagree.
- Double-check that the row and column order of the leadfield covariance matches your EEG channel order (e.g., channel index 0 in both tensors corresponds to C1); misalignment silently degrades cleaning quality.
- The pipeline enforces even epoch lengths. If the requested epoch and sampling rate yield an odd sample count, GEDAI pads before processing and trims afterward.
- When running on GPU, move both EEG data and leadfield tensors to the target device prior to calling the API.
- Enable `verbose_timing=True` during development to gather profiling markers such as `start_batch`, `modwt_analysis`, and `batch_done`.
- If you only require cleaned signals, set `skip_checks_and_return_cleaned_only=True` to avoid collecting diagnostic metadata.

---

## Citation

Return of the GEDAI: Unsupervised EEG Denoising based on Leadfield Filtering (2025) [bioRxiv]. https://doi.org/10.1101/2025.10.04.680449

Ros, T., Férat, V., Huang, Y., Colangelo, C., Kia, S. M., Wolfers, T., Vulliemoz, S., & Michela, A.

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
