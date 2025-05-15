import subprocess


patients = ["Pat_02", "Pat_03"]
phases   = ["rsPre", "taskLearn", "taskTest", "rsPost"]
bands    = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]

for patient in patients:
    sample_rate = 2048 if patient == "Pat_02" else 1024
    for phase in phases:
        for band in bands:
            cmd = (
                # f'slanzarv -m 4096 --jobname "BRAINNET" --nomail '
                f'python src/corr_matrix_band.py '
                f'--patient {patient} '
                f'--phase {phase} '
                f'--band {band} '
                f'--sample_rate {sample_rate}'
                f' --plot_all'
            )
            print(cmd)
            subprocess.call((cmd), shell=True)
            # os.system(cmd)