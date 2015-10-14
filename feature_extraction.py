from scipy.fftpack import fft
import numpy as np


eps = 0.00000001

def signal_energy(frame):
    """
    This function computes energy of signal from audio file.

    ARGUMENTS
        frame: a 1d array obtained from reading audio.wav file

    RETURNS
        signal energy, a scalar value
    """

    return np.sum(frame ** 2) / np.float64(len(frame))

def signal_energy_entropy(frame, num_of_sub_frames=10):
    """
    This function computes entropy of energy of signal.

    ARGUMENTS
        frame:              a 1d array obtained from reading audio.wav file
        num_of_sub_frames:  an int value

    RETURNS
        entropy:    entropy of energy, a scalar value

    """
    frame_energy = np.sum(frame ** 2)    # total frame energy
    frame_len = len(frame)
    sub_frame_len = int(np.floor(frame_len / num_of_sub_frames))
    if frame_len != sub_frame_len * num_of_sub_frames:
            frame = frame[0:sub_frame_len * num_of_sub_frames]

    sub_frame_matrix = frame.reshape(sub_frame_len, num_of_sub_frames, order='F').copy()

    # Compute normalized sub-frame energies:
    sub_frame_energy_component = np.sum(sub_frame_matrix ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(sub_frame_energy_component * np.log2(np.abs(sub_frame_energy_component + eps)))
    return entropy

def spectral_centroid_and_spread(X, fs):
    # Computes spectral centroid of frame (given abs(FFT) and sampling frequency)
    f_x = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    weighted_magnitude = Xt / Xt.max()
    numerator = np.sum(f_x * weighted_magnitude)
    denominator = np.sum(weighted_magnitude) + eps

    # Centroid:
    centroid = (numerator / denominator)

    # Spread:
    spread = np.sqrt(np.sum(((f_x - centroid) ** 2) * weighted_magnitude) / denominator)

    # Normalize:
    centroid_normalized = centroid / (fs / 2.0)
    spread_normalized = spread / (fs / 2.0)

    return centroid_normalized, spread_normalized

def spectral_entropy(X, num_of_sub_spectra=10):
    """Computes the spectral entropy"""
    spectral_len = len(X)                         # number of frame samples
    total_spectral_energy = np.sum(X ** 2)            # total spectral energy

    sub_spectral_len = int(np.floor(spectral_len / num_of_sub_spectra))   # length of sub-frame
    if spectral_len != sub_spectral_len * num_of_sub_spectra:
        X = X[0:sub_spectral_len * num_of_sub_spectra]

    sub_spectra = X.reshape(sub_spectral_len, num_of_sub_spectra, order='F').copy()  # define sub-frames (using matrix reshape)
    sub_spectral_energies = np.sum(sub_spectra ** 2, axis=0) / (total_spectral_energy + eps)   # compute spectral sub-energies
    entropy = -np.sum(sub_spectral_energies * np.log2(sub_spectral_energies + eps))       # compute spectral entropy

    return entropy

def spectral_roll_off(X, c, fs):
    """Computes spectral roll-off"""
    total_spectral_energy = np.sum(X ** 2)
    spectral_len = len(X)
    threshold = c * total_spectral_energy
    # Find the frequency position where respective spectral energy is equal to c*totalEnergy
    cumulative_sum = np.cumsum(X ** 2) + eps
    [a, ] = np.nonzero(cumulative_sum > threshold)
    if len(a) > 0:
        mC = np.float64(a[0]) / (float(spectral_len))
    else:
        mC = 0.0
    return mC

def audio_features(frame, fs):
    """
    This function combines all the features from audio file into an array
    :param frame: an array, read from audio.wav file
    :param fs: sampling rate, from audio.wav file
    :return: audio_feature_matrix: a 1d array of features
    """

    X = abs(fft(frame))
    nfft = int(len(frame) / 2)
    X = X[0:nfft]
    X = X / len(X)
    # setting threshold value
    c = 0.90
    audio_feature_matrix = np.array([[signal_energy(frame), signal_energy_entropy(frame, num_of_sub_frames=10),
                               spectral_centroid_and_spread(X, fs)[0], spectral_centroid_and_spread(X, fs)[1],
                               spectral_entropy(X, num_of_sub_spectra=10), spectral_roll_off(X, c, fs)]])
    return audio_feature_matrix

def mode_feature(mode):
    """
    This function converts mode in string format to a numeric array element.
    :param mode: 'hand' or 'table', a string
    :return: array element
    """
    mode_to_feature = {'hand': -1, 'table': 1}
    return np.matrix(mode_to_feature[mode])

def create_feature_vector(touch_features, frame, fs, mode):
    """
    This function combines together all the features from touch.csv, audio.wav files and mode
    :param touch_features: a data frame, read from touch.csv using pandas.read_csv
    :param frame: a 1d array of signal from audio.wav file
    :param fs: sampling rate
    :param mode: an array element (1 or -1)
    :return: 1d array, feature vector
    """
    features = np.hstack((touch_features.values, audio_features(frame, fs), mode_feature(mode)))
    return features
