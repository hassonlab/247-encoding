import glob
import os

import numpy as np
from scipy.io import loadmat
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def trim_signal(signal):
    bin_size = 32  # 62.5 ms (62.5/1000 * 512)
    signal_length = signal.shape[0]

    if signal_length < bin_size:
        print("Ignoring conversation: Small signal")
        return None

    cutoff_portion = signal_length % bin_size
    if cutoff_portion:
        signal = signal[:-cutoff_portion, :]

    return signal


def detrend_signal(mat_signal): # Detrending

    y = mat_signal
    X = np.arange(len(y)).reshape(-1,1)
    pf = PolynomialFeatures(degree=2)
    Xp = pf.fit_transform(X)

    model = LinearRegression()
    model.fit(Xp, y)
    trend = model.predict(Xp)
    mat_signal = y - trend

    return mat_signal

def fake_signal(stitch, convo_id):

    mat_len = stitch[convo_id]-stitch[convo_id-1] # mat file length
    mat_signal = np.empty((mat_len, 1))
    mat_signal.fill(np.nan)

    return mat_signal


def load_electrode_data(args, elec_id, stitch, z_score):
    '''Loads specific electrodes mat files
    '''
    if args.project_id == 'tfs':
        DATA_DIR = '/projects/HASSON/247/data/conversations-car'
        process_flag = 'preprocessed'
    elif args.project_id == 'podcast':
        DATA_DIR = '/projects/HASSON/247/data/podcast-data'
        process_flag = 'preprocessed_all'
    else:
        raise Exception('Invalid Project ID')

    convos = sorted(glob.glob(os.path.join(DATA_DIR, str(args.sid), 'NY*Part*conversation*')))

    all_signal = []
    missing_convos = []
    for convo_id, convo in enumerate(convos, 1):
        if args.conversation_id != 0 and convo_id != args.conversation_id:
            continue

        file = glob.glob(os.path.join(convo, process_flag, '*_' + str(elec_id) + '.mat'))

        if len(file) == 1: # conversation file exists
            file = file[0]

            mat_signal = loadmat(file)['p1st']
            mat_signal = mat_signal.reshape(-1, 1)

            if mat_signal is None:
                continue

            mat_signal = detrend_signal(mat_signal) # detrend conversation signal
            if z_score: # doing erp
                mat_signal = stats.zscore(mat_signal)

        elif len(file) == 0: # conversation file does not exist
            if args.sid != 7170:
                raise SystemExit(f'Error: Conversation file does not exist for electrode {elec_id} at {convo}')
            missing_convos.append(os.path.basename(convo)) # append missing convo name
            mat_signal = fake_signal(stitch, convo_id)

        else: # more than 1 conversation files
            raise SystemExit(f'Error: More than 1 signal file exists for electrode {elec_id} at {convo}')

        all_signal.append(mat_signal) # append conversation signal

    if args.project_id == 'tfs':
        elec_signal = np.vstack(all_signal)
    else:
        elec_signal = np.array(all_signal)

    return elec_signal, missing_convos