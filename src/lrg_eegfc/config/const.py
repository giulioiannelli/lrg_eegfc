BRAIN_BANDS = {
    'delta':      (0.53,   4),
    'theta':      (4,      8),
    'alpha':      (8,     13),
    'beta':       (13,    30),
    'low_gamma':  (30,    80),
    'high_gamma': (80,   300),
}

phase_labels = ['rsPre', 'taskLearn', 'taskTest', 'rsPost']
param_keys_list = ['fs', 'fcutHigh', 'fcutLow', 'filter_order', 'NotchFilter', 'DataDimensions']