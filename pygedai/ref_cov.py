"""
This module requires additional installations to function properly.
Interpolate reference covariance matrix provided by https://github.com/neurotuning/GEDAI-master.
"""
import re
import torch
from pathlib import Path
from importlib.resources import files, as_file

try:
    import pandas as pd
except:
    raise ImportError("pandas is required for this function. Please install it via 'pip install pandas'.")
try:
    import numpy as np
except:
    raise ImportError("numpy is required for this function. Please install it via 'pip install numpy'.")
try:
    from scipy.spatial import cKDTree
except:
    raise ImportError("scipy is required for this function. Please install it via 'pip install scipy'.")
try:
    import mat73
except:
    raise ImportError("mat73 is required for this function. Please install it via 'pip install mat73'.")

REQUIRED_COLUMNS = ["channel_name", "X", "Y", "Z"]

def interpolate_ref_cov(electrode_positions, 
                        dtype=torch.float64) -> torch.Tensor:
    """
    Interpolate the reference covariance matrix based on provided electrode positions.
    """
    if not isinstance(electrode_positions, pd.DataFrame):
        raise ValueError("electrode_positions must be a pandas DataFrame")
    
    if not set(REQUIRED_COLUMNS).issubset(electrode_positions.columns):
        raise ValueError(f"electrode_positions must contain the following columns: {REQUIRED_COLUMNS}")
    
    res = files("pygedai").joinpath("data", "fsavLEADFIELD_4_GEDAI.mat")
    with as_file(res) as p: # gives a real filesystem path even from a wheel
        M = mat73.loadmat(str(p))
    gram = np.asarray(L["gram_matrix_avref"]) # [channels x channels]
    L = M["leadfield4GEDAI"]
    leadfield_electrodes = L["electrodes"]

    selected_channels = _select_channels(leadfield_electrodes, electrode_positions)

    gram_sel  = _get_leadfield_selected(L, gram, leadfield_electrodes, selected_channels)
    ref_cov = torch.from_numpy(np.asarray(gram_sel, np.float64))

    print("Reference covariance matrix shape:", ref_cov.shape)
    print("Reference covariance matrix dtype:", ref_cov.dtype)
    print("Warning: This reference covariance is based on a template and may not perfectly match your electrode configuration.")
    print(f"WARNING: Ensure the electrode positions by row in 'electrode_positions' match the order expected by your data processing pipeline (row 0 in 'electrode_positions' = {selected_channels[0]} with this is assumed that the EEG data torch tensor at row 0 is {selected_channels[0]}).")

    return ref_cov.to(dtype=dtype)

def _select_channels(electrodes, chdf):
    tgt = [
        d for d in electrodes
        if d.get("Type") == "EEG"
        and d.get("Loc") is not None
        and not str(d["Name"]).endswith("h")
    ]
    tgt_names = np.array([d["Name"] for d in tgt], dtype=object)
    tgt_xyz = np.vstack([np.asarray(d["Loc"], float) for d in tgt])

    # build E-names + scale your XYZ to target radius
    src = chdf.copy()
    src.insert(0, "row", np.arange(len(src))) # row id for sanity (0..N-1)

    src_xyz = src[["X","Y","Z"]].to_numpy(float)
    src_xyz = src_xyz / np.linalg.norm(src_xyz, axis=1, keepdims=True)
    src_xyz = src_xyz * np.median(np.linalg.norm(tgt_xyz, axis=1))

    # nearest-neighbor mapping (outputs are in the same order as src_xyz input)
    tree = cKDTree(tgt_xyz)
    dists, idxs = tree.query(src_xyz, k=1)
    src["mapped_name"] = tgt_names[idxs]
    src["dist_mm"] = dists * 1000.0

    # exact label matches override
    tgt_name_set = set(tgt_names.tolist())
    mask_exact = src["channel_name"].astype(str).isin(tgt_name_set)
    src.loc[mask_exact, "mapped_name"] = src.loc[mask_exact, "channel_name"].astype(str)
    src.loc[mask_exact, "dist_mm"] = 0.0

    print("median distance (mm):", float(np.median(src["dist_mm"])))
    print()  # first 25 rows, same order

    selected_channels = src["mapped_name"].tolist() # Contains all mapped names in order

    print("First 5 channel mappings:")
    print(src.loc[:, ["row","channel_name","mapped_name","dist_mm"]].head())

    return selected_channels

def _norm(lbl: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", lbl.lower())

def _get_leadfield_selected(L, gram, electrodes, selected_channels):
    # extract template labels
    if isinstance(electrodes, (list, tuple)):
        template_labels = [e["Name"] if isinstance(e, dict) else str(e) for e in electrodes]
    elif isinstance(electrodes, dict) and "Name" in electrodes:
        names = electrodes["Name"]
        template_labels = list(names) if isinstance(names, (list, tuple)) else [str(names)]
    else:
        template_labels = list(map(str, electrodes))

    lut = {_norm(t): i for i, t in enumerate(template_labels)}
    idx = np.array([lut.get(_norm(ch), -1) for ch in selected_channels], dtype=int)
    if (idx < 0).any():
        missing = [ch for ch, i in zip(selected_channels, idx) if i < 0]
        raise ValueError(f"Electrode labels not found in template: {missing}")

    gram_sel = gram[np.ix_(idx, idx)]
    #labels_sel = [template_labels[i] for i in idx.tolist()]
    return gram_sel