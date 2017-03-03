"""Code to simulate the decoding of anticipation in an oddball experiment.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import matplotlib.pyplot as plt


def trial(amplitude=100, trial_tmin=-300,
          trial_tmax=500, sampling_frequency=250,
          t_peak=250, t_sigma=60,
          noise_level=100,
          return_time=False):
    """Naive generator/simulator of trials.
    """
    t = np.arange(trial_tmin, trial_tmax, 1000.0/sampling_frequency)
    signal = amplitude * np.exp(-(t - t_peak)**2 / (2.0 * t_sigma**2))
    noise = np.random.normal(loc=0.0, scale=noise_level, size=t.size)
    if return_time:
        return t, signal + noise
    else:
        return signal + noise


if __name__ == '__main__':
    np.random.seed(0)
    trial_tmin = -300  # ms
    trial_tmax = 500  # ms
    sampling_frequency = 250  # Hz
    amplitude_standard = 100
    amplitude_deviant = 150
    t_peak = 250  # ms
    t_sigma = 60  # ms
    noise_level = 100
    n_trial_standard = 50
    n_trial_deviant = 50
    n_trial_deviant_test = 30
    window_size = 50  # ms
    clf = LinearDiscriminantAnalysis()
    n_folds = 10

    t, s = trial(amplitude=amplitude_standard,
                 trial_tmin=trial_tmin,
                 trial_tmax=trial_tmax,
                 sampling_frequency=sampling_frequency,
                 t_peak=t_peak,
                 t_sigma=t_sigma,
                 noise_level=noise_level,
                 return_time=True)  # generate 1 trial

    print("Generate %s standard trials" % n_trial_standard)
    ss_standard = np.array([trial(amplitude=amplitude_standard,
                                  trial_tmin=trial_tmin,
                                  trial_tmax=trial_tmax,
                                  sampling_frequency=sampling_frequency,
                                  t_peak=t_peak,
                                  t_sigma=t_sigma,
                                  noise_level=noise_level) for i in range(n_trial_standard)])

    print("Generate %s deviant trials" % n_trial_deviant)
    ss_deviant = np.array([trial(amplitude=amplitude_deviant,
                                 trial_tmin=trial_tmin,
                                 trial_tmax=trial_tmax,
                                 sampling_frequency=sampling_frequency,
                                 t_peak=t_peak,
                                 t_sigma=t_sigma,
                                 noise_level=noise_level) for i in range(n_trial_deviant)])

    print("Generate %s deviant trials for testing" % n_trial_deviant_test)
    ss_deviant_test = np.array([trial(amplitude=amplitude_deviant,
                                      trial_tmin=trial_tmin,
                                      trial_tmax=trial_tmax,
                                      sampling_frequency=sampling_frequency,
                                      t_peak=t_peak,
                                      t_sigma=t_sigma,
                                      noise_level=noise_level) for i in range(n_trial_deviant_test)])

    # number of timepoints in the decoding time window:
    window_size_tp = int(window_size / 1000.0 * sampling_frequency)
    # containers of results:
    accuracy_standard_vs_deviant = np.zeros(t[:-window_size_tp].size)
    accuracy_deviant = np.zeros((t[:-window_size_tp].size,
                                 t[:-window_size_tp].size))
    print("Estimating accuracy of decoding standard vs deviant at different time lags...")
    print("... and estimating accuracy of deviant_test at different time lags")
    for i, t_start_train in enumerate(t[:-window_size_tp]):
        timestep_window = np.logical_and(t >= t_start_train,
                                         t < t_start_train + window_size)
        X = np.vstack([ss_standard[:, timestep_window],
                       ss_deviant[:, timestep_window]])
        y = np.concatenate([np.zeros(n_trial_deviant),
                            np.ones(n_trial_deviant)])
        scores = cross_val_score(clf, X, y, cv=n_folds)
        accuracy_standard_vs_deviant[i] = scores.mean()
        clf.fit(X, y)
        for j, t_start_test in enumerate(t[:-window_size_tp]):
            timestep_window_test = np.logical_and(t >= t_start_test,
                                                  t < t_start_test + window_size)
            X_test = ss_deviant_test[:, timestep_window_test]
            y_test = np.ones(n_trial_deviant_test)
            accuracy_deviant[i, j] = (clf.predict(X_test) == y_test).mean()

    plt.interactive(True)
    plt.figure()
    plt.plot(t, s, label='one trial')
    plt.plot(t, ss_standard.mean(0), label='standard (mean)')
    plt.plot(t, ss_deviant.mean(0), label='deviant (mean)')
    plt.plot(t, ss_deviant_test.mean(0), label='deviant_test (mean)')
    plt.title("One trial and Evoked Potentials (averages)")
    plt.legend()

    plt.figure()
    t_plot = t[window_size_tp / 2:-window_size_tp / 2]
    plt.plot(t_plot, accuracy_standard_vs_deviant)
    plt.title("Decoding accuracy of standard vs deviant across time")

    plt.figure()
    plt.imshow(accuracy_deviant, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.title("Cross-decoding accuracy of deviant_test across time")
