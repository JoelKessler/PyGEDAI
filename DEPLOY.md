# Local Testing

## Create DIST
python setup.py sdist bdist_wheel

## Install Locally
conda create -n pygedai python=3.12 -y # For e.g. older intel mac
conda activate pygedai
pip install mne
pip install "torch==2.2.2"
pip install "numpy==1.26.4"
pip install dist/pygedai-0.1.0-py3-none-any.whl --force-reinstall

# Verify Local Environment works
python - <<'PY'
import torch, numpy as np; print("torch:", torch.__version__, "numpy:", np.__version__)
PY
No errors should be thrown.

## Install from pip
pip install mne 
pip install "numpy==1.26.4"
pip install "pygedai[torch]"

## Test Locally
python -c """
import pathlib
import torch
import mne
import numpy as np
from pygedai import gedai

root = pathlib.Path.cwd()
raw_filepath = str(root / 'testing' / 'samples' / 'with_artifacts' / 'artifact_jumps.set')
print(raw_filepath)
raw = mne.io.read_raw_eeglab(raw_filepath, preload=True)
raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
eeg = torch.from_numpy(raw.get_data(picks='eeg'))

leadfield_filepath = str(root / 'testing' / 'leadfield_calibrated' / 'leadfield4GEDAI_eeg_61ch.npy')
leadfield = torch.from_numpy(np.load(leadfield_filepath))

result = gedai(eeg, sfreq=raw.info['sfreq'], leadfield=leadfield)

cleaned = result['cleaned'].detach().cpu().numpy()

print('cleaned shape:', cleaned.shape)
print('SENSAI score:', float(result['sensai_score']))
"""