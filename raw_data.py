import shutil
import numpy as np
import pandas as pd
import os
import yaml
from os import listdir
from os.path import isfile, join
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py


class RawData:
    def __init__(self):
        self.types: np.ndarray | None = None
        self.stages: np.ndarray | None = None
        self.chains: np.ndarray | None = None
        self.tags: np.ndarray | None = None
        self.parameters: np.ndarray | None = None
        self.observations: np.ndarray | None = None
        self.weights: np.ndarray | None = None

        self.no_chains = 0
        self.no_stages = 0
        self.no_samples = 0

    def _load_one_csv(self, file_sample, path_data, no_parameters):
        path_samples = os.path.join(path_data, file_sample)

        # Check if the file exists and is not empty
        if not os.path.exists(path_samples) or os.path.getsize(path_samples) == 0:
            print(f"Failed to load file {file_sample} because it is not present or empty.")
            # Return empty arrays with appropriate shapes
            return (np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.empty((0, no_parameters), dtype=np.float64),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64).reshape((-1, 1)))

        # read file in chunks
        chunk_iter = pd.read_csv(path_samples, chunksize=10000, header=None)
        # process chunks and concatenate
        chunks = [chunk for chunk in chunk_iter]
        df_samples = pd.concat(chunks)

        N_samples = len(df_samples)

        idx = np.array(df_samples.iloc[:, 0].map({"accepted": 0, "prerejected": 1, "rejected": 2}), dtype=int)
        stages = int(file_sample[3]) * np.ones(N_samples, dtype=int)
        chains = int(file_sample[file_sample.find("rank") + 4:file_sample.find(".")]) * \
            np.ones(N_samples, dtype=int)
        parameters = np.array(df_samples.iloc[:, 1:1 + no_parameters], dtype=np.float64)
        tags = np.array(df_samples.iloc[:, 1 + no_parameters].map({0: 0, 1: 1}).fillna(-1), dtype=int)
        observation = np.array(df_samples.iloc[:, 2 + no_parameters:], dtype=np.float64)

        # compute weights
        # prerejected and rejected have weight=0
        widx = idx == 0
        temp = np.arange(N_samples)[widx]
        weights = np.zeros(N_samples)
        if len(temp) > 0:
            temp[:-1] -= temp[1:]
            temp[-1] -= N_samples
            temp = -temp
            weights[widx] = temp
        # if sum(widx) > 0:
        weights = weights.reshape((-1, 1))
        return (idx, stages, chains, tags, parameters, observation, weights)

    def load_from_folder(self, target_folder, config_name="config_mcmc_bayes", num_paralel_processes=8, verbose=False):
        path_config = os.path.join(target_folder, 'common_files/' + config_name + '.yaml')
        path_data = os.path.join(target_folder, 'saved_samples/' + config_name + '/raw_data')

        # load yaml file
        with open(path_config, 'r') as stream:
            conf = yaml.safe_load(stream)

        no_parameters = conf["no_parameters"]

        file_samples = [f for f in listdir(path_data) if isfile(join(path_data, f))]
        file_samples.sort()

        types_all = []
        stages_all = []
        chains_all = []
        tags_all = []
        parameters_all = []
        observations_all = []
        weights_all = []

        if num_paralel_processes > 1:
            with ProcessPoolExecutor(max_workers=num_paralel_processes) as executor:
                # iterate over files in directory
                future_to_file = {executor.submit(self._load_one_csv, f, path_data,
                                                  no_parameters): f for f in file_samples}
                for future in as_completed(future_to_file):
                    single_res = future.result()
                    types_all.append(single_res[0])
                    stages_all.append(single_res[1])
                    chains_all.append(single_res[2])
                    tags_all.append(single_res[3])
                    parameters_all.append(single_res[4])
                    observations_all.append(single_res[5])
                    weights_all.append(single_res[6])
        else:
            for f in file_samples:
                single_res = self._load_one_csv(f, path_data, no_parameters)
                types_all.append(single_res[0])
                stages_all.append(single_res[1])
                chains_all.append(single_res[2])
                tags_all.append(single_res[3])
                parameters_all.append(single_res[4])
                observations_all.append(single_res[5])
                weights_all.append(single_res[6])

        self.types = np.concatenate(types_all)
        self.stages = np.concatenate(stages_all)
        self.chains = np.concatenate(chains_all)
        self.tags = np.concatenate(tags_all)
        self.parameters = np.concatenate(parameters_all)
        self.observations = np.concatenate(observations_all)
        self.weights = np.concatenate(weights_all)
        if len(self.types) > 0:
            self.no_stages = max(self.stages) + 1
            self.no_chains = max(self.chains) + 1
            self.no_samples = len(self.types)

        if verbose:
            print("raw data: no_stages", self.no_stages)
            print("raw data: no_chains", self.no_chains)
            print("raw data: no_samples", self.no_samples)
            print("raw data: no_nonconverging", len(self.types[self.tags < 0]))
            print("raw data: p", np.shape(self.parameters))
            print("raw data: w", np.shape(self.weights))
            # print(self.weights[:20])
            # print(self.weights[-20:])
            # print(self.weights[self.weights>0][:])
            print("raw_data: np.sum(weights):", np.sum(self.weights))

    def load(self, folder_samples, no_parameters, no_observations, verbose=False):
        folder_samples = os.path.join(folder_samples, 'raw_data')
        file_samples = [f for f in listdir(folder_samples) if isfile(join(folder_samples, f))]
        file_samples.sort()
        N = len(file_samples)

        self.types = np.empty((0, 1), dtype=np.int8)
        self.stages = np.empty((0, 1), dtype=np.int8)
        self.chains = np.empty((0, 1), dtype=np.int8)
        self.tags = np.empty((0, 1), dtype=np.int8)
        self.parameters = np.empty((0, no_parameters))
        self.observations = np.empty((0, no_observations))
        self.weights = np.empty((0, 1), dtype=np.int64)

        for i in range(N):
            path_samples = folder_samples + "/" + file_samples[i]
            df_samples = pd.read_csv(path_samples, header=None)

            types = df_samples.iloc[:, 0]
            idx = np.zeros(len(types), dtype=np.int8)
            # idx[types == "accepted"] = 0
            idx[types == "prerejected"] = 1
            idx[types == "rejected"] = 2
            # print(np.shape(self.types))
            # print(np.shape(idx))

            stg = int(file_samples[i][3])
            self.no_stages = max(self.no_stages, stg + 1)
            stages = stg * np.ones(len(types), dtype=np.int8)

            chain = int(file_samples[i][file_samples[i].find("rank") + 4:file_samples[i].find(".")])
            self.no_chains = max(self.no_chains, chain + 1)
            chains = chain * np.ones(len(types), dtype=np.int8)

            parameters = np.array(df_samples.iloc[:, 1:1 + no_parameters])
            tags = np.array(df_samples.iloc[:, 1 + no_parameters])
            observation = np.array(df_samples.iloc[:, 2 + no_parameters:])

            self.types = np.append(self.types, idx)
            self.stages = np.append(self.stages, stages)
            self.chains = np.append(self.chains, chains)
            self.tags = np.append(self.tags, tags)
            self.parameters = np.vstack((self.parameters, parameters))
            self.observations = np.vstack((self.observations, observation))

            # compute weights
            # prerejected and rejected have weight=0
            widx = np.ones(len(types), dtype=bool)
            widx[types == "prerejected"] = False
            widx[types == "rejected"] = False
            # not considering first sample
            # TODO get rid of last value
            temp = np.arange(len(types))[widx]
            temp = np.append(temp, len(types))
            temp = np.diff(temp)
            weights = np.zeros(len(types))
            weights[widx] = temp
            # if sum(widx) > 0:
            weights = weights.reshape((-1, 1))
            self.weights = np.vstack((self.weights, weights)).astype(int)

        self.no_samples = len(self.types)
        if verbose:
            print("raw data: no_stages", self.no_stages)
            print("raw data: no_chains", self.no_chains)
            print("raw data: no_samples", self.no_samples)
            print("raw data: no_nonconverging", len(self.types[self.tags < 0]))
            print("raw data: p", np.shape(self.parameters))
            print("raw data: w", np.shape(self.weights))
            # print(self.weights[:20])
            # print(self.weights[-20:])
            # print(self.weights[self.weights>0][:])
            print("raw_data: np.sum(weights):", np.sum(self.weights))

    def len(self):
        assert self.types is not None
        return len(self.types)

    def no_parameters(self):
        assert self.parameters is not None
        return np.shape(self.parameters)[1]

    def no_observations(self):
        assert self.observations is not None
        return np.shape(self.observations)[1]

    def filter(self, types=None, stages=None, chains=None, tags=None):
        assert self.tags is not None
        assert self.types is not None
        assert self.stages is not None
        assert self.chains is not None
        assert self.parameters is not None
        assert self.observations is not None
        assert self.weights is not None

        idx = np.ones(len(self.types), dtype=bool)

        if types is not None:
            idx *= np.isin(self.types, types)
        if stages is not None:
            idx *= np.isin(self.stages, stages)
        if chains is not None:
            idx *= np.isin(self.chains, chains)
        if tags is not None:
            idx *= np.isin(self.tags, tags)

        raw_data = RawData()
        raw_data.types = self.types[idx]
        raw_data.tags = self.tags[idx]
        raw_data.stages = self.stages[idx]
        raw_data.chains = self.chains[idx]
        raw_data.parameters = self.parameters[idx]
        raw_data.observations = self.observations[idx]
        raw_data.weights = self.weights[idx]

        if len(raw_data.types) > 0:
            raw_data.no_stages = max(raw_data.stages) + 1
            raw_data.no_chains = max(raw_data.chains) + 1
            raw_data.no_samples = len(raw_data.types)
        return raw_data

    def save_hdf5(self, file_path):
        assert self.types is not None
        assert self.stages is not None
        assert self.tags is not None
        assert self.chains is not None
        assert self.parameters is not None
        assert self.observations is not None
        assert self.weights is not None

        with h5py.File(file_path, 'w') as f:
            f.create_dataset('types', data=self.types)
            f.create_dataset('stages', data=self.stages)
            f.create_dataset('chains', data=self.chains)
            f.create_dataset('tags', data=self.tags)
            f.create_dataset('parameters', data=self.parameters)
            f.create_dataset('observations', data=self.observations)
            f.create_dataset('weights', data=self.weights)
            f.create_dataset('no_all', data=np.array([self.no_stages, self.no_chains, self.no_samples], dtype=int))

    def load_from_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as file:
            self.types: np.ndarray | None = file['types'][:]  # type: ignore
            self.stages: np.ndarray | None = file['stages'][:]  # type: ignore
            self.chains: np.ndarray | None = file['chains'][:]  # type: ignore
            self.tags: np.ndarray | None = file['tags'][:]  # type: ignore
            self.parameters: np.ndarray | None = file['parameters'][:]  # type: ignore
            self.observations: np.ndarray | None = file['observations'][:]  # type: ignore
            self.weights: np.ndarray | None = file['weights'][:]  # type: ignore
            no_all = file['no_all'][:]  # type: ignore
            self.no_stages: int = no_all[0]  # type: ignore
            self.no_chains: int = no_all[1]  # type: ignore
            self.no_samples: int = no_all[2]  # type: ignore

    def print_statistics(self):
        assert self.types is not None
        assert self.stages is not None
        assert self.tags is not None
        assert self.chains is not None
        assert self.parameters is not None
        assert self.observations is not None
        assert self.weights is not None
        print("raw data: no_stages", self.no_stages)
        print("raw data: no_chains", self.no_chains)
        print("raw data: no_samples", self.no_samples)
        print("raw data: no_nonconverging", len(self.types[self.tags < 0]))
        print("raw data: p", np.shape(self.parameters))
        print("raw data: w", np.shape(self.weights))
        print("raw_data: np.sum(weights):", np.sum(self.weights))
        all_sizes = {}
        all_sizes["types"] = self.types.shape
        all_sizes["stages"] = self.stages.shape
        all_sizes["chains"] = self.chains.shape
        all_sizes["tags"] = self.tags.shape
        all_sizes["parameters"] = self.parameters.shape
        all_sizes["observations"] = self.observations.shape
        all_sizes["weights"] = self.weights.shape
        print("raw_data: all_sizes:", all_sizes)


class MultiRawData:
    def __init__(self):
        self.run_ids: np.ndarray | None = None
        self.run_id_map: dict[int, str] | None = None
        self.types: np.ndarray | None = None
        self.stages: np.ndarray | None = None
        self.chains: np.ndarray | None = None
        self.tags: np.ndarray | None = None
        self.parameters: np.ndarray | None = None
        self.parameters_normalized: np.ndarray | None = None
        self.observations: np.ndarray | None = None
        self.weights: np.ndarray | None = None

        self.no_samples = 0

    def build_from_list(self, list_of_raw_data, list_of_configs, names=None):
        run_ids = []
        run_id_map = {}
        types = []
        stages = []
        chains = []
        tags = []
        parameters = []
        parameters_normalized = []
        observations = []
        weights = []

        for i, raw_data in enumerate(list_of_raw_data):
            run_ids.append(i * np.ones(raw_data.no_samples, dtype=int))
            if names is not None:
                run_id_map[i] = names[i]
            else:
                run_id_map[i] = str(i)

            conf = list_of_configs[i]

            # transform data according to conf['transformations']
            all_parameters = raw_data.parameters
            parameters_transformed = raw_data.parameters.copy()

            for i in range(parameters_transformed.shape[1]):
                tmp = np.log(all_parameters[:, i])  # log transform
                tmp = (tmp - conf['transformations'][i]['options']['mu'])  # substract mean
                tmp = tmp / conf['transformations'][i]['options']['sigma']  # divide by std
                parameters_transformed[:, i] = tmp

            types.append(raw_data.types)
            stages.append(raw_data.stages)
            chains.append(raw_data.chains)
            tags.append(raw_data.tags)
            parameters.append(raw_data.parameters)
            parameters_normalized.append(parameters_transformed)
            observations.append(raw_data.observations)
            weights.append(raw_data.weights)

        nums_obs = min([obs.shape[1] for obs in observations])
        observations = [obs[:, :nums_obs] for obs in observations]

        self.run_ids = np.concatenate(run_ids)
        self.run_id_map = run_id_map
        self.types = np.concatenate(types)
        self.stages = np.concatenate(stages)
        self.chains = np.concatenate(chains)
        self.tags = np.concatenate(tags)
        self.parameters = np.concatenate(parameters)
        self.parameters_normalized = np.concatenate(parameters_normalized)
        self.observations = np.concatenate(observations)
        self.weights = np.concatenate(weights)

        self.no_samples = len(self.types)

    def load_from_folders_hdf5(self, folders_list):
        list_of_raw_data = []
        list_of_configs = []
        names = []
        for folder in folders_list:
            # find hdf5 file in folder
            path_to_file = os.path.join(folder, "raw_data.hdf5")
            path_to_config = os.path.join(folder, "common_files", "config_mcmc_bayes.yaml")

            if not (os.path.exists(path_to_file)) and (os.path.exists(path_to_config)):
                # if no hdf5 file found, skip folder
                print("No hdf5 file or config found in folder ", folder)
                continue

            raw_data = RawData()
            raw_data.load_from_hdf5(path_to_file)
            list_of_raw_data.append(raw_data)

            # names as name of the last folder in path
            names.append(path_to_file.split("/")[-2])

            # load and save config
            with open(path_to_config) as f:
                conf = yaml.safe_load(f)
                list_of_configs.append(conf)
        self.build_from_list(list_of_raw_data, list_of_configs, names)

    def filter(self, types=None, stages=None, chains=None, tags=None, run_ids=None):
        assert self.tags is not None
        assert self.run_ids is not None
        assert self.types is not None
        assert self.stages is not None
        assert self.chains is not None
        assert self.parameters is not None
        assert self.parameters_normalized is not None
        assert self.observations is not None
        assert self.weights is not None

        idx = np.ones(len(self.types), dtype=bool)

        if types is not None:
            idx *= np.isin(self.types, types)
        if stages is not None:
            idx *= np.isin(self.stages, stages)
        if chains is not None:
            idx *= np.isin(self.chains, chains)
        if tags is not None:
            idx *= np.isin(self.tags, tags)
        if run_ids is not None:
            idx *= np.isin(self.run_ids, run_ids)

        raw_data = MultiRawData()
        raw_data.types = self.types[idx]
        raw_data.tags = self.tags[idx]
        raw_data.stages = self.stages[idx]
        raw_data.chains = self.chains[idx]
        raw_data.parameters = self.parameters[idx]
        raw_data.parameters_normalized = self.parameters_normalized[idx]
        raw_data.observations = self.observations[idx]
        raw_data.weights = self.weights[idx]
        raw_data.run_ids = self.run_ids[idx]
        raw_data.run_id_map = self.run_id_map

        raw_data.no_samples = len(raw_data.types)
        return raw_data

    def print_statistics(self):
        assert self.types is not None
        assert self.stages is not None
        assert self.tags is not None
        assert self.chains is not None
        assert self.parameters is not None
        assert self.parameters_normalized is not None
        assert self.observations is not None
        assert self.weights is not None
        assert self.run_ids is not None

        print("raw data: no_samples", self.no_samples)
        print("raw data: no_nonconverging", len(self.types[self.tags < 0]))
        print("raw data: p", np.shape(self.parameters))
        print("raw data: w", np.shape(self.weights))
        print("raw_data: np.sum(weights):", np.sum(self.weights))
        all_sizes = {}
        all_sizes["types"] = self.types.shape
        all_sizes["stages"] = self.stages.shape
        all_sizes["chains"] = self.chains.shape
        all_sizes["tags"] = self.tags.shape
        all_sizes["parameters"] = self.parameters.shape
        all_sizes["observations"] = self.observations.shape
        all_sizes["parameters_normalized"] = self.parameters_normalized.shape
        all_sizes["weights"] = self.weights.shape
        all_sizes["run_ids"] = self.run_ids.shape
        print("raw_data: all_sizes:", all_sizes)

