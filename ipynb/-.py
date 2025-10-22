#!/home/opisthofulax/anaconda3/envs/lapbrain/bin/python
# coding: utf-8

# # Ultrametric Distance (Permutation Robust): Patient Comparison
# 
# Compute permutation-robust ultrametric distances across phases for every patient, per EEG band.
# Plots show, for each band, permutation-robust distance matrices side-by-side for all patients.
# 
# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import lrgsglib
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))  # Test plot to ensure matplotlib is working

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
#
from lrgsglib import *
#
move_to_rootf(pathname='lrg_eegfc')
#
from lrg_eegfc import *
# Figure output path
path_figs = Path('data') / 'figures' / 'new_ultrametric_permutation_robust_pat_comparison'
path_figs.mkdir(parents=True, exist_ok=True)
#
phase_labels = list(PHASE_LABELS)
bands = list(BRAIN_BANDS.keys())
patients = list(PATIENTS_LIST)
#
correlation_protocol = dict(filter_type='abs', spectral_cleaning=False)
#
print(f'Patients: {patients}')
print(f'Phases: {phase_labels}')
print(f'Bands: {bands}')


# ## Load Data

# In[ ]:


data_dict, int_label_map = load_data_dict()
print('✓ Data loaded')


# ## Build Ultrametric and Linkage Structures per Patient, Phase, Band

# In[ ]:


def compute_structures_for_patient(patient: str, filter_order: int = 1):
    U = {b: {} for b in bands}
    Z = {b: {} for b in bands}
    D = {b: {} for b in bands}

    pin_labels = int_label_map[patient]['label']

    for phase in phase_labels:
        data_pat_phase = data_dict[patient][phase]
        data_pat_phase_ts = data_pat_phase['data']
        fs_raw = data_pat_phase.get('fs', None)
        fs = float(np.asarray(fs_raw).flat[0]) if fs_raw is not None else 1000.0

        for band in bands:
            G, label_dict, lnkgM, clTh, corr_mat, dists = process_network_for_phase(
                data_pat_phase_ts, fs, band, correlation_protocol, pin_labels, filter_order=filter_order
            )
            if lnkgM is None or G is None:
                U[band][phase] = None
                Z[band][phase] = None
                D[band][phase] = None
                continue
            try:
                U[band][phase] = extract_ultrametric_matrix(lnkgM, G.number_of_nodes())
            except Exception as e:
                print(f'[WARN] {patient} {phase} {band}: failed to extract ultrametric ({e})')
                U[band][phase] = None
            Z[band][phase] = lnkgM
            if isinstance(dists, dict):
                D[band][phase] = dists.get('condensed')
            else:
                D[band][phase] = None
    return U, Z, D

ultra_by_pat = {}
linkage_by_pat = {}
condensed_by_pat = {}
for pat in patients:
    print(f'Computing ultrametric structures for {pat} ...')
    U, Z, D = compute_structures_for_patient(pat)
    ultra_by_pat[pat] = U
    linkage_by_pat[pat] = Z
    condensed_by_pat[pat] = D
print('✓ Ultrametric structures computed for all patients')


# ## Phase Distance Matrices per Patient, Band

# In[ ]:


metric = 'euclidean'
results = {band: {} for band in bands}
n = len(phase_labels)

for band in bands:
    for pat in patients:
        dm = np.full((n, n), np.nan, dtype=float)
        pin_labels = int_label_map[pat]['label']
        for i, pi in enumerate(phase_labels):
            for j, pj in enumerate(phase_labels):
                if i == j:
                    dm[i, j] = 0.0
                    continue
                Z1 = linkage_by_pat[pat][band].get(pi)
                Z2 = linkage_by_pat[pat][band].get(pj)
                D1 = condensed_by_pat[pat][band].get(pi)
                D2 = condensed_by_pat[pat][band].get(pj)
                if Z1 is None or Z2 is None or D1 is None or D2 is None:
                    dm[i, j] = np.nan
                else:
                    dm[i, j] = ultrametric_distance_permutation_robust(Z1, Z2, D1, D2, pin_labels, metric=metric)
        results[band][pat] = dm
    print(f'Band {band}: ✓ done')

print('✓ All phase-distance matrices computed')


# ## Plot: Side-by-Side Patients per Band

# In[ ]:


def plot_band_side_by_side_permutation(band: str):
    vmax = 0.0
    for pat in patients:
        m = results[band].get(pat)
        if m is not None and np.isfinite(m).any():
            vmax = max(vmax, np.nanmax(m))
    if vmax == 0.0:
        print(f'[SKIP] No finite values for {band}')
        return

    fig, axes = plt.subplots(1, len(patients), figsize=(4*len(patients)+2, 4), constrained_layout=True)
    if len(patients) == 1:
        axes = [axes]

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray')

    for ax, pat in zip(axes, patients):
        M = results[band][pat]
        im = ax.imshow(M, vmin=0.0, vmax=vmax, cmap=cmap, aspect='equal')
        ax.set_title(f"{pat}")
        ax.set_xticks(range(len(phase_labels)))
        ax.set_yticks(range(len(phase_labels)))
        ax.set_xticklabels(phase_labels, rotation=45, ha='right')
        ax.set_yticklabels(phase_labels)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label('Permutation-robust distance')
    outdir = path_figs / band
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / 'permutation_robust.png'
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.show()
    print(f'Saved: {outfile}')

for band in bands:
    plot_band_side_by_side_permutation(band)

print('✓ All figures generated')


# In[ ]:




