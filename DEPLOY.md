# Local Testing

## Create DIST
python setup.py sdist bdist_wheel

## Install Locally
pip install dist/pygedai-0.1.0-py3-none-any.whl

## Test Locally
python -c "import pathlib, numpy as np, torch, mne; from GEDAI import gedai; root = pathlib.Path.cwd(); raw = mne.io.read_raw_eeglab(str(root / 'samples' / 'with_artifacts' / 'artifact_jumps.set'), preload=True); matlab_cleaned_raw = mne.io.read_raw_eeglab(str(root / 'samples' / 'matlab_cleaned' / 'cleaned_artifact_jumps.set'), preload=True); raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False); matlab_cleaned_raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False); eeg = torch.from_numpy(raw.get_data(picks='eeg')); matlab_cleaned = matlab_cleaned_raw.get_data(picks='eeg'); leadfield = torch.from_numpy(np.load(root / 'leadfield_calibrated' / 'leadfield4GEDAI_eeg_61ch.npy')); result = gedai(eeg, sfreq=raw.info['sfreq'], leadfield=leadfield); cleaned = result['cleaned'].detach().cpu().numpy(); print('cleaned shape:', cleaned.shape); print('SENSAI score:', float(result['sensai_score'])); print('max abs diff vs matlab:', float(np.max(np.abs(cleaned - matlab_cleaned))))"