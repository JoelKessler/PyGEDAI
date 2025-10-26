Verify

python -c "from verify_gedai import compare_matlab_vs_python_gedai; compare_matlab_vs_python_gedai(matlab_cleaned_set='./samples/cleaned_artifact_jump.set',noisy_raw_set='./samples/artifact_jumps.set',leadfield='./samples/leadfield4GEDAI_61ch.npy', denoising_strength='auto',epoch_size=1.0, wavelet_bands=9, r_tol=0.999,rel_rmse_tol=1e-3, line_noise_hz=50.0)"