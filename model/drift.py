from scipy.stats import ks_2samp


def detect_drift(old_data, new_data, threshold=0.05):
    stat, p_val = ks_2samp(old_data, new_data)
    return p_val < threshold