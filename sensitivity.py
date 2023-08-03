import os
import sys

import ruamel.yaml as yaml
import numpy as np
import pandas as pd

n_samples_per_diff = 33


def collect_data(config_bayes, sensitivity_dir):
    files = os.listdir(sensitivity_dir)
    param_files = sorted([f for f in files if "params_" in f and ".csv" in f])
    data_files = sorted([f for f in files if "output_" in f and ".csv" in f])
    assert len(param_files) == len(data_files)

    no_parameters = config_bayes["no_parameters"]
    no_observations = config_bayes["no_observations"]

    parameters = np.zeros((0, no_parameters))
    observations = np.zeros((0, no_observations))
    for i, (pf, df) in enumerate(zip(param_files, data_files)):
        print("Reading parameters from CSV: ", pf)
        p_samples = pd.read_csv(os.path.join(sensitivity_dir, pf), header=0)
        d_samples = pd.read_csv(os.path.join(sensitivity_dir, df))
        n_collected = len(d_samples)

        # sort
        # obs_data = obs_data[observations_part[:, 0].argsort()]
        # cut unfinished samples at the end (not all diffs computed)
        cut = n_collected - int(n_collected / n_samples_per_diff) * n_samples_per_diff
        obs_data = np.array(d_samples.iloc[:-cut,1:])
        n_collected = obs_data.shape[0]

        # get corresponding parameters
        params = np.array(p_samples.iloc[:n_collected, :])

        observations = np.vstack((observations, obs_data))
        parameters = np.vstack((parameters, params))

    print(observations.shape)
    print(parameters.shape)
    return parameters, observations


def compute_differences(parameters, observations):
    tidx = 10
    npar = parameters.shape[1]
    n_points = int(parameters.shape[0]/n_samples_per_diff)

    diff_close = np.zeros((n_points, npar))
    diff_far = np.zeros((n_points, npar))
    points = np.zeros((n_points, npar))

    for i in range(n_points):
        sub_p = parameters[i*n_samples_per_diff:(i+1)*n_samples_per_diff]
        sub_o = observations[i*n_samples_per_diff:(i+1)*n_samples_per_diff, tidx]
        # diff nominator
        diff_o_close = sub_o[1:npar+1] + sub_o[npar+1:2*npar+1] - 2*sub_o[0]
        diff_o_far = sub_o[2*npar+1:3*npar+1] + sub_o[3*npar+1:] - 2*sub_o[0]
        # diff denominator
        diff_p_close = np.diag(sub_p[1:npar+1,:]) - np.diag(sub_p[npar+1:2*npar+1,:])
        diff_p_far = np.diag(sub_p[2*npar+1:3*npar+1,:]) - np.diag(sub_p[3*npar+1:,:])

        points[i] = sub_p[0]
        diff_close[i] = diff_o_close / diff_p_close**2
        diff_far[i] = diff_o_far / diff_p_far**2

    return points, diff_close, diff_far


def plot_diff(config_bayes, points, diffs, name):
    npoints, npar = points.shape
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(npar, npar, sharex=False, sharey=False, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    trans = config_bayes["transformations"]
    for j in range(npar):
        axis = axes[0, j]
        # determine parameter name
        label = trans[j]["name"]
        axis.set_title(label, x=0.5, rotation=45, multialignment='center')

    for i in range(npar):
        for j in range(npar):
            axis = axes[i, j]
            if i==j:
                axis.plot(points[:,i], diffs[:,i], '.')
            elif i>j:
                axis.scatter(points[:,i], points[:,j], c=diffs[:,i], cmap='hsv')
            else:
                axis.scatter(points[:,i], points[:,j], c=diffs[:,j], cmap='hsv')

    # fig.savefig(os.path.join(sensitivity_dir, name + ".pdf"), bbox_inches="tight")


def read_sensitivity_config(work_dir):
    # test if config exists, copy from rep_dir if necessary
    sens_config_file = os.path.join(work_dir, "config_sensitivity.yaml")

    if not os.path.exists(sens_config_file):
        raise Exception("Main configuration file 'config.yaml' not found in workdir.")

    # read config file
    with open(sens_config_file, "r") as f:
        sens_config_dict = yaml.safe_load(f)

    sens_config_dict["script_dir"] = os.path.dirname(os.path.abspath(__file__))
    sens_config_dict["rep_dir"] = os.path.abspath(os.path.join(sens_config_dict["script_dir"], "../../.."))
    return sens_config_dict


def read_config(output_dir):
    # create and cd workdir
    rep_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = output_dir

    # Files in the directory are used by each simulation at that level
    common_files_dir = os.path.join(work_dir, "common_files")

    # test if config exists, copy from rep_dir if necessary
    config_file = os.path.join(work_dir, "config.yaml")
    if not os.path.exists(config_file):
        # to enable processing older results
        config_file = os.path.join(common_files_dir, "config.yaml")
        if not os.path.exists(config_file):
            raise Exception("Main configuration file 'config.yaml' not found in workdir.")
        else:
            import warnings
            warnings.warn("Main configuration file 'config.yaml' found in 'workdir/common_files'.",
                          category=DeprecationWarning)

    # read config file and setup paths
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict["work_dir"] = work_dir
    config_dict["script_dir"] = rep_dir

    config_dict["common_files_dir"] = common_files_dir
    config_dict["bayes_config_file"] = os.path.join(common_files_dir,
                                                    config_dict["surrDAMH_parameters"]["config_file"])
    
    return config_dict


# if __name__ == "__main__":

#     # default parameters
#     output_dir = None

#     len_argv = len(sys.argv)
#     assert len_argv > 1, "Specify output dir & number of processes & number of best fits"
#     if len_argv > 1:
#         output_dir = os.path.abspath(sys.argv[1])

#     # setup paths and directories
#     config_dict = read_config(output_dir)
#     with open(config_dict["bayes_config_file"]) as f:
#         config_bayes = yaml.safe_load(f)

#     sens_config_dict = read_sensitivity_config(output_dir)

#     sensitivity_dir = os.path.join(output_dir, "sensitivity")
#     par, obs = collect_data(sensitivity_dir)
#     points, diff_close, diff_far = compute_differences(par, obs)

#     plot_diff(points, diff_close, "diff_close")
#     plot_diff(points, diff_far, "diff_far")
