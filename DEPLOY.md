# Local Testing

## Create DIST
python setup.py sdist bdist_wheel

## Install Locally
conda create -n pygedai python=3.12 -y # For e.g. older intel mac
conda activate pygedai
pip install "torch==2.2.2"
pip install dist/pygedai-0.1.0-py3-none-any.whl --force-reinstall
pip install mne

## Install from pip
pip install "pygedai[torch]"
pip install mne

## Test Locally
python -c """
import pathlib
import torch
import mne
from pygedai import gedai

root = pathlib.Path.cwd()
raw = mne.io.read_raw_eeglab(str(root / 'samples' / 'with_artifacts' / 'artifact_jumps.set'), preload=True)
raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
eeg = torch.from_numpy(raw.get_data(picks='eeg'))

leadfield = torch.from_numpy(np.load(root / 'leadfield_calibrated' / 'leadfield4GEDAI_eeg_61ch.npy'))

result = gedai(eeg, sfreq=raw.info['sfreq'], leadfield=leadfield)

cleaned = result['cleaned'].detach().cpu().numpy()

print('cleaned shape:', cleaned.shape)
print('SENSAI score:', float(result['sensai_score']))
"""