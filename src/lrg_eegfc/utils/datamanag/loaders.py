from ...shared import *

def load_mat_pat_data(patient: str, phase: str, mat_path: Path):
    try:
        tmp_mat = loadmat(str(mat_path / patient / f'{phase}.mat'))
    except Exception as e:
        logging.warning(f"{type(e).__name__} loading {patient} {phase}: {e}. Using h5py fallback.")
        f = h5py.File(mat_path / patient / f'{phase}.mat', 'r')
        tmp_mat = {k: np.array(v) for k, v in f.items()}
    return tmp_mat

def load_data_dict(
    mat_path: Path,
    pat_list: List[str],
    phase_labels: List[str],
    param_keys_list: List[str]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame]]:
    data_dict: Dict[str, Dict[str, Any]] = {}
    int_label_pos_map: Dict[str, pd.DataFrame] = {}

    for pat in pat_list:
        data_dict[pat] = {}
        patnum = int(pat.split('_')[-1])
        patpath = mat_path / pat

        ch_dat   = pd.read_csv(patpath / f'Implant_pat_{patnum}.csv')
        ch_names = pd.read_csv(patpath / 'channel_labels.csv')
        int_label_pos_map[pat] = ch_names.merge(ch_dat, on='label', how='left')

        df = int_label_pos_map[pat][['x', 'y', 'z']]
        df = df.apply(lambda s: s.str.replace(',', '.', regex=False).astype(float))

        for phase in phase_labels:
            data_dict[pat][phase] = {}
            tmp_mat = load_mat_pat_data(pat, phase, mat_path)
            arr = tmp_mat['Data']
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T
            data_dict[pat][phase]['data'] = arr

            try:
                for param in tmp_mat['Parameters'].dtype.names:
                    if param in param_keys_list:
                        data_dict[pat][phase][param] = tmp_mat['Parameters'][param][0][0][0][0]
            except Exception:
                print(f'No parameters found for {pat} {phase}', end='\r', flush=True)

    return data_dict, int_label_pos_map