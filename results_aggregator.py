import os
import shutil
import sys
from raw_data import RawData


def compare_files(file_list):
    """
    Compare list of files whether they are binary equal.
    Auxiliary testing function.
    """
    for f in file_list:
        for g in file_list:
            import filecmp
            if not (os.path.exists(f) and os.path.exists(g)):
                print("on of the files does not exist", f, g)
            elif not filecmp.cmp(f,g, shallow=False):
                print("not equal", f, g)


def copy_configs(directories, copy_list, output_dir):
    """
    Copy files in copy_list from all given directories to a single dir.
    Auxiliary testing function.
    """
    dest_dir = os.path.join(script_dir, output_dir)
    for output_dir in directories:
        basename = os.path.basename(output_dir)
        # if not os.path.exists(dest_dir):
            # os.makedirs(dest_dir)

        for spath in copy_list:
            src = os.path.join(output_dir, spath)
            if os.path.exists(src):
                dst = os.path.join(dest_dir, basename + "_" + os.path.basename(spath))
                print(src)
                shutil.copy(src, dst)


def list_dir(dirpath, dir=True):
    """
    Returns list of fullpaths to all directories (dir=True) or files (dir=False)
    in directory given by `dirpath`.
    """
    # next(os.walk(dirpath))[1] : walk is generator that gives [0]-root, [1]-dirnames, [2]-filenames
    s = 1 if dir else 2
    dirs = [os.path.join(dirpath,d) for d in next(os.walk(dirpath))[s]]
    return dirs


def get_subfolder_paths(directory):
    subfolder_paths = []

    for folder_path in list_dir(directory):
        for subfolder_path in list_dir(folder_path):
            if 'saved_samples' in os.listdir(subfolder_path):
                subfolder_paths.append(subfolder_path)

    return subfolder_paths


def get_subfolder_paths_processed(directory):
    subfolder_paths = []

    for folder_path in list_dir(directory):
        for subfolder_path in list_dir(folder_path):
            if 'raw_data.hdf5' in os.listdir(subfolder_path):
                subfolder_paths.append(subfolder_path)

    return subfolder_paths


def replicate_folder_structure(subfolders, original_path, new_path):
    # Get the length of the original path. We will use this to properly replicate the folder structure
    length_of_original = len(original_path)

    subfolders_new_location = []
    for subfolder in subfolders:
        # Get the relative path of the subfolder by removing the original path from it
        relative_path = subfolder[length_of_original:]

        # Remove any leading slashes
        relative_path = relative_path.lstrip('/')

        # Create the corresponding folder in the new directory
        new_folder_path = os.path.join(new_path, relative_path)
        subfolders_new_location.append(new_folder_path)
        os.makedirs(new_folder_path, exist_ok=True)
    return subfolders_new_location


def copy_files(source_dir, target_dir, copy_list):
    """
    Copy files and dirs from `source_dir` into `target_dir`.
    Files to copy is given relatively in `copy_list`.
    if path is a directory, full tree is copied.
    if path is a file, the file is copied keeping the relative dir structure.
    """
    for spath in copy_list:
        src = os.path.join(source_dir, spath)
        dst = os.path.join(target_dir, spath)
        print(src)
        if os.path.isdir(src):
            # if dir given, copy it as whole
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            # if file given, create subfolder structure first
            dstfolder = os.path.dirname(dst)
            if not os.path.exists(dstfolder):
                os.makedirs(dstfolder)
            shutil.copy(src, dst)
        else:
            raise Exception("Not a file nor directory. " + src)


def create_hdf(src_folder, dst_folder, drop_prerejected=False):
    """
    For all given simulation folders in `subfolders` it reads raw data and saves it into HDF5.
    """
    try:
        raw_data_loaded = RawData()
        raw_data_loaded.load_from_folder(src_folder)
        if drop_prerejected:
            raw_data_loaded = raw_data_loaded.filter(types=[0, 2])
        raw_data_loaded.save_hdf5(os.path.join(dst_folder, "raw_data.hdf5"))
    except Exception as err:
        print(err)


def download_copy_files(subfolders, subfolders_new_location, copy_list=None, drop_prerejected=False, verbose=True):
    """
    Copy simulation files from subfolders to subfolders_new_location. All paths are assumed to be absolute.
    `copy_list` contains files/dirs to be copied selectively.
    Tweak - `config.yaml` is searched automatically in two places: root and  in `common_files`
    due to older simulations.
    """
    if copy_list is None:
        copy_list = ["common_files/config_mcmc_bayes.yaml",
                     "common_files/A04_hm_tmpl.yaml",
                    #  "saved_samples/config_mcmc_bayes/raw_data",
                     "saved_samples/config_mcmc_bayes/output.yaml"]
    
    for src_folder, dst_folder in zip(subfolders, subfolders_new_location):
        if verbose:
            print("Processing folder: ", src_folder)
        
        cp = copy_list.copy()
        # resolve older simulations with different path of config.yaml
        if os.path.exists(os.path.join(src_folder, "config.yaml")):
            cp.append("config.yaml")
        elif os.path.exists(os.path.join(src_folder, "common_files/config.yaml")):
            cp.append("common_files/config.yaml")
        else:
            raise Exception("'config.yaml' not found in " + src_folder)

        try:
            copy_files(src_folder, dst_folder, cp)
            create_hdf(src_folder, dst_folder, drop_prerejected)
        except Exception as err:
            print(err)


def filter_folders_to_empty(subfolders, subfolders_new_location):
    if len(subfolders) != len(subfolders_new_location):
        # raise ValueError
        raise ValueError("The number of subfolders and their new locations must be the same")

    # Reverse loop to allow removal of items without breaking the indexing
    for i in range(len(subfolders) - 1, -1, -1):
        path = subfolders_new_location[i]
        if os.path.isdir(path) and not os.listdir(path):  # Check if the path is a directory and it's empty
            continue
        else:
            # If it's not an empty directory, remove its entry from both lists
            del subfolders[i]
            del subfolders_new_location[i]

    return subfolders, subfolders_new_location


# if __name__ == "__main__":

#     # TESTING
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     sim_dirs = []

#     # len_argv = len(sys.argv)
#     # assert len_argv > 1, "Specify simulation dirs (dir or file with paths)!"
#     # if len_argv > 1:
#     #     if os.path.isdir(sys.argv[1]):
#     #         sim_dirs = os.listdir(sys.argv[1])
#     #     elif os.path.isfile(sys.argv[1]):
#     #         filepath = os.path.abspath(sys.argv[1])
#     #         with open(filepath) as f:
#     #             sim_dirs = f.readlines()
    
#     dir_path = "/home/paulie/Workspace/endorse_results/borehole_H1"
#     sim_dirs = [os.path.join(dir_path, p) for p in os.listdir(dir_path)]

#     flow123d_templates = [os.path.join(p, "common_files", "A04_hm_tmpl.yaml") for p in sim_dirs]
#     compare_files(flow123d_templates)

#     os.mkdir("configs")
#     copy_configs(sim_dirs, ["config.yaml"], "configs")
    

    

