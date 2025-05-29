BRAIN_BANDS = {
    'delta':      (0.53,   4),
    'theta':      (4,      8),
    'alpha':      (8,     13),
    'beta':       (13,    30),
    'low_gamma':  (30,    80),
    'high_gamma': (80,   300),
}
BRAIN_BANDS_LABELS = {
    'delta':      r'$\delta$',
    'theta':      r'$\theta$',
    'alpha':      r'$\alpha$',
    'beta':       r'$\beta$',
    'low_gamma':  r'$\gamma_{\rm l}$',
    'high_gamma': r'$\gamma_{\rm h}$',
}

phase_labels = ['rsPre', 'taskLearn', 'taskTest', 'rsPost']
param_keys_list = ['fs', 'fcutHigh', 'fcutLow', 'filter_order', 'NotchFilter', 'DataDimensions']