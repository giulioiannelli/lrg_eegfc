import argparse
import logging
from pathlib import Path

import numpy as np
import scipy.io
import h5py
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from lrgsglib.core import *
from lrg_eegfc.core import *

from lrgsglib.config.funcs import bandpass_sos
from lrg_eegfc.core import *
from lrg_eegfc.utils.parsers.corr_matrix_band import get_parser, BRAIN_BANDS


def load_data(patient: str, phase: str, mat_path: Path):
    try:
        tmp_mat = scipy.io.loadmat(str(mat_path / patient / f'{phase}.mat'))
    except Exception as e:
        logging.warning(f"{type(e).__name__} loading {patient} {phase}: {e}. Using h5py fallback.")
        f = h5py.File(mat_path / patient / f'{phase}.mat', 'r')
        tmp_mat = {k: np.array(v) for k, v in f.items()}
    data = tmp_mat['Data']
    if data.shape[0] > data.shape[1]:
        data = data.T
    return data


def compute_and_save(patient: str, phase: str, band: str, filttime: int,
                     sample_rate: float, filter_order: int):
    mat_path = Path('data') / 'stereoeeg_patients'
    out_dir = Path('data') / 'correlations' / patient
    per_band_corr = Path('data') / '250414_preanalysis' / 'per_band_corr' / patient
    savepth = out_dir / f'{band}_{phase}_corr.npy'
    if os.path.exists(savepth):
        C_filt = np.load(savepth)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        per_band_corr.mkdir(parents=True, exist_ok=True)

        log_file = out_dir / f'{phase}_{band}.log'
        logging.basicConfig(filename=str(log_file), level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info(f"Starting computation for {patient} {phase} {band}")

        data = load_data(patient, phase, mat_path)
        if filttime > 0:
            data = data[:, :filttime]
        filt = bandpass_sos(data, *BRAIN_BANDS[band], sample_rate, filter_order)
        logging.info(f"Data shape after trim: {data.shape}")
        
        C_filt = np.abs(np.corrcoef(filt))
        C_filt, eigvals_C, eigvecs_C, lambda_min, lambda_max, signal_mask = clean_correlation_matrix(C_filt)
        logging.info(f"Cleaned correlation matrix: eigenvalues range [{lambda_min}, {lambda_max}]")
        np.fill_diagonal(C_filt, 0)
        G_filt = nx.from_numpy_array(C_filt)
        logging.info(f"Computed filtered correlation for band {band}")
        try:
            Th, Einf, Pinf = compute_threshold_stats(G_filt)
            Pinf_diff = np.diff(Pinf)
            jumps = np.where(Pinf_diff != 0)[0]
            th = Th[jumps[1]]
            logging.info(f"Threshold stats: Th={th} Jumps at {jumps}")
        except Exception as e:
            logging.warning(f"Error computing threshold stats: {e}")
            th = 0.5
            logging.info(f"Using default threshold: {th}")
        C_filt[C_filt < th] = 0
        np.save(savepth, C_filt)
    G_filt = nx.from_numpy_array(C_filt)
    return C_filt, G_filt

def plot_corr_matrix(matrix: np.ndarray, patient: str, phase: str, band: str):
    out_dir = Path('data') / '250414_preanalysis' / 'per_band_corr' / patient
    plt.figure()
    plt.imshow(matrix, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f'{patient} {phase} {band}')
    plt.savefig(out_dir / f'{phase}_{band}_corr.pdf')
    plt.close()
    logging.info("Saved correlation plot")

def plot_entropy(G: nx.Graph, steps: int, patient: str, phase: str, band: str):
    network_entropy = entropy(G, t1=-3, t2=5, steps=steps)
    Sm1 = network_entropy[0] / network_entropy[0].max()
    speC = network_entropy[1] / network_entropy[1].max()
    tau_scale = network_entropy[-1]

    out_dir = Path('data') / '250414_preanalysis' / 'per_band_corr' / patient
    plt.figure()
    plt.plot(tau_scale, 1 - Sm1, label=r'$1-S$')
    plt.plot(tau_scale[:-1], speC, label=r'$C$')
    plt.xlabel('Scale (τ)')
    plt.ylabel('Normalized Values')
    plt.legend()
    plt.title(f'Entropy Metrics {patient} {phase} {band}')
    plt.savefig(out_dir / f'{phase}_{band}_entropy.pdf')
    plt.close()
    logging.info("Saved entropy plot")


def plot_entropy(G: nx.Graph, steps: int, patient: str, phase: str, band: str):
    network_entropy = entropy(G, t1=-3, t2=5, steps=steps)
    Sm1 = network_entropy[0] / network_entropy[0].max()
    speC = network_entropy[1] / network_entropy[1].max()
    tau_scale = network_entropy[-1]

    out_dir = Path('data') / '250414_preanalysis' / 'per_band_corr' / patient
    plt.figure()
    plt.plot(tau_scale, 1 - Sm1, label=r'$1-S$')
    plt.plot(tau_scale[:-1], speC, label=r'$C$')
    plt.xlabel('Scale (τ)')
    plt.ylabel('Normalized Values')
    plt.legend()
    plt.title(f'Entropy Metrics {patient} {phase} {band}')
    plt.xscale('log')
    plt.savefig(out_dir / f'{phase}_{band}_entropy.pdf')
    plt.close()
    logging.info("Saved entropy plot")


def plot_dendrogram(linkage_matrix, labels, threshold, ch_int_name_map, patient: str, phase: str, band: str):
    relabel = [ch_int_name_map[n] for n in labels]
    out_dir = Path('data') / '250414_preanalysis' / 'per_band_corr' / patient
    fig, ax = plt.subplots(figsize=(5, 15))
    dendro = dendrogram(linkage_matrix, ax=ax,
                         labels=relabel,
                         color_threshold=threshold,
                         above_threshold_color='k',
                         leaf_font_size=9,
                         orientation='right')
    ax.axvline(threshold, color='b', linestyle='--',
               label=r'$\mathcal{D}_{\rm th}$')
    ax.legend()
    ax.set_xscale('log')
    tmin = linkage_matrix[::, 2][0] - 0.2*linkage_matrix[::, 2][0]
    tmax = linkage_matrix[::, 2][-1] + 0.1*linkage_matrix[::, 2][-1]
    ax.set_xlim(tmin,tmax)
    plt.title(f'Dendrogram {patient} {phase} {band}')
    plt.savefig(out_dir / f'{phase}_{band}_dendrogram.pdf')
    fig.tight_layout
    plt.close()
    logging.info("Saved dendrogram plot")
    return dendro

def plot_graph(GG: nx.Graph, dendro: dict, patient: str, phase: str, band: str, ch_int_name_map: dict):
    out_dir = Path('data') / '250414_preanalysis' / 'per_band_corr' / patient
    fig, ax = plt.subplots()
    leaf_label_colors = {label: color for label, color in zip(
        dendro['ivl'], dendro['leaves_color_list'])}
    relabel_list = dendro['ivl']
    node_colors = [leaf_label_colors[label] for label in relabel_list]
    widths = [GG[u][v]['weight'] for u, v in GG.edges()]
    ch_int_name_map_ = {k: ch_int_name_map[k] for k in list(GG.nodes())}
    pos = nx.spring_layout(GG, seed=5)
    nx.draw(GG, pos=pos, ax=ax, node_size=80, font_size=10,
            width=widths, node_color=node_colors,
            alpha=0.7, with_labels=True, labels=ch_int_name_map_)
    plt.title(f'Network Graph {patient} {phase} {band}')
    plt.savefig(out_dir / f'{phase}_{band}_graph.pdf')
    plt.close()
    logging.info("Saved graph plot")

def main():
    mat_path = Path('data') / 'stereoeeg_patients'
    ch_names = loadmat(mat_path / 'ChannelNames.mat')
    ch_names = [name[0] for name in ch_names['ChannelNames'][0]]
    ch_int_name_map = {i: name for i, name in enumerate(ch_names)}
    parser = get_parser()
    args = parser.parse_args()
    if args.plot_all:
        args.plot_corr_mat = True
        args.plot_entropy = True
        args.plot_dendrogram = True
        args.plot_graph = True
    C_clean, G = compute_and_save(
        args.patient, args.phase, args.band,
        args.filttime, args.sample_rate, args.filter_order
    )
    if args.plot_corr_mat:
        plot_corr_matrix(C_clean, args.patient, args.phase, args.band)
    if args.plot_entropy:
        plot_entropy(G, args.entropy_steps, args.patient, args.phase, args.band)
    dendro = None
    if args.plot_dendrogram or args.plot_graph:
        spec, L, rho, Trho, tau = compute_laplacian_properties(G)
        dists = squareform(Trho)
        Z, labels, tmax = compute_normalized_linkage(dists, G)
        th, *_ = compute_optimal_threshold(Z)
    if args.plot_dendrogram:
        dendro = plot_dendrogram(Z, labels, th, ch_int_name_map,
                                 args.patient, args.phase, args.band)
    if args.plot_graph and dendro is not None:
        plot_graph(G, dendro,
                   args.patient, args.phase, args.band, ch_int_name_map)

if __name__ == '__main__':
    main()