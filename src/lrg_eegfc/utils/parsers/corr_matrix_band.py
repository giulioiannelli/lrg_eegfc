import argparse
from pathlib import Path

phase_labels = ['rsPre', 'taskLearn', 'taskTest', 'rsPost']
BRAIN_BANDS = {
    'delta':      (0.53,   4),
    'theta':      (4,      8),
    'alpha':      (8,     13),
    'beta':       (13,    30),
    'low_gamma':  (30,    80),
    'high_gamma': (80,   300),
}

def get_parser():
    parser = argparse.ArgumentParser(
        description='Compute correlation matrices, entropy, dendrogram, and graph for SEEG data'
    )
    parser.add_argument('--patient', '-p', type=str, required=True,
                        help='Patient identifier (e.g. Pat_01)')
    parser.add_argument('--phase', '-ph', type=str, required=True,
                        choices=phase_labels, help='Recording phase')
    parser.add_argument('--band', '-b', type=str, required=True,
                        choices=list(BRAIN_BANDS.keys()), help='Frequency band')
    parser.add_argument('--filttime', '-ft', type=int, default=-1,
                        help='Max time index to use (-1 for full length)')
    parser.add_argument('--sample_rate', '-sr', type=float, default=2048.0,
                        help='Sampling rate for bandpass filter')
    parser.add_argument('--filter_order', '-fo', type=int, default=4,
                        help='Filter order for bandpass filter')
    parser.add_argument('--plot_corr_mat', '-pcm', action='store_true',
                        help='Generate and save correlation matrix plot')
    parser.add_argument('--plot_entropy', '-pe', action='store_true',
                        help='Compute and plot network entropy')
    parser.add_argument('--entropy_steps', '-es', type=int, default=400,
                        help='Number of steps for entropy calculation')
    parser.add_argument('--plot_dendrogram', '-pd', action='store_true',
                        help='Compute and plot dendrogram clustering')
    parser.add_argument('--plot_graph', '-pg', action='store_true',
                        help='Compute and plot network graph')
    parser.add_argument('--plot_all', action='store_true',
                        help='Enable all plots')
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)

