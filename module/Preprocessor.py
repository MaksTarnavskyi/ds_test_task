import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import logging
import os
from functools import partial
import getopt, sys

cpu_count = max(cpu_count() - 1, 1)

class Preprocessor:
    def __init__(self):

        self._stats_features = {
            "2": {
                "mean": None,
                "std": None
                # "min": None,
                # "max": None
            }
        }

    #------------------------ Fit train -----------------------------------------------------------#

    def fit_train(self, train_filename="data/train.tsv", chunksize=500, sep="\t",n_jobs=cpu_count):
        """
        Read train file by chunks, split data for each core, run calculations for statistics
        :param train_filename: path for train file
        :param chunksize: amount of rows per chunck
        :param sep: separator between column values in train.tsv
        :param n_jobs: amount of cpus to use
        :return: None
        """
        all_result_maps = []

        for chunk in pd.read_csv(train_filename, chunksize=chunksize, sep=sep):
            chunk_split = np.array_split(chunk.values, n_jobs)
            pool = Pool(n_jobs)
            result_map = pool.map(self._preprocess_chunk_for_train, chunk_split)
            pool.close()
            pool.join()

            all_result_maps.extend(result_map)

        result_df = pd.DataFrame(all_result_maps)
        self._calculate_statistics(result_df)

    @staticmethod
    def _get_feature_type(feature_string):
        return feature_string.split(",")[0]

    @staticmethod
    def _get_feature_count(feature_string):
        return len(feature_string.split(",")) -1

    @staticmethod
    def _get_float_feature_values(feature_string):
        return np.array(feature_string.split(",")).astype(float)

    def _preprocess_chunk_for_train(self, chunk_split):
        all_features = []
        all_features_squared = []
        count_of_rows = 0

        for row in chunk_split:
            try:
                id_job, values_in_string = row
                float_features = self._get_float_feature_values(values_in_string)[1:]

                all_features.append(float_features)
                all_features_squared.append(np.power(float_features, 2))
                count_of_rows += 1

            except Exception as e:
                logging.error(str(e)+"row: "+str(row))

        feature_type = self._get_feature_type(feature_string=chunk_split[0][1])

        chunk_features_sum = np.array(all_features).sum(axis=0)
        chunk_features_squared_sum = np.array(all_features_squared).sum(axis=0)
        # chunk_features_min = np.array(all_features).min(axis=0)
        # chunk_features_max = np.array(all_features).max(axis=0)

        return {"chunk_features_sum": chunk_features_sum,
                "chunk_features_squared_sum": chunk_features_squared_sum,
                # "chunk_features_min" : chunk_features_min,
                # "chunk_features_max" : chunk_features_max
                "count_of_rows": count_of_rows,
                "feature_type": feature_type}

    def _calculate_statistics(self, result_df):
        int_features_sum = result_df["chunk_features_sum"].values.sum(axis=0)
        int_features_squared_sum = result_df["chunk_features_squared_sum"].values.sum(axis=0)
        count_of_rows = result_df["count_of_rows"].values.sum(axis=0)

        feature_type = str(result_df["feature_type"].values[0])

        features_mean = int_features_sum / count_of_rows
        features_std = np.sqrt(int_features_squared_sum / count_of_rows - np.power(features_mean, 2))

        self._stats_features[feature_type]["mean"] = features_mean
        self._stats_features[feature_type]["std"] = features_std

        # features_min = result_df["chunk_features_min"].values.min(axis=0)
        # features_max = result_df["chunk_features_max"].values.max(axis=0)
        # self._stats_features[feature_type]["min"] = features_min
        # self._stats_features[feature_type]["max"] = features_max

    #--------------------------- Process test -------------------------------------------------------

    def process_test(self, test_filename="data/test.tsv", output_folder="result", output_filename="test_proc.tsv", chunksize=500, sep="\t", n_jobs=cpu_count, process_method="standardization"):
        """
        Read test file by chunks, split data for each core, run process function for each row, save rows in output file
        :param test_filename: path to test.tsv - data for processing
        :param output_folder: path for output folder (module creates it, if it doesn't exist)
        :param output_filename: name for output file
        :param chunksize: amount of rows per chunck
        :param sep: separator between column values in test.tsv
        :param n_jobs: amount of cpus to use
        :param process_method: method for process features
        :return: None
        """
        append = False #for creating file and later appending

        self._check_path(output_folder) #check if exists output_folder
        full_output_filename = os.path.join(output_folder, output_filename) #full path

        for chunk in pd.read_csv(test_filename, chunksize=chunksize, sep=sep):
            chunk_split = np.array_split(chunk.values, n_jobs)

            pool = Pool(n_jobs)
            result_map = pool.map(partial(self._process_chunk_for_test,process_method=process_method),chunk_split)

            pool.close()
            pool.join()

            for df in result_map:
                if append:
                    df.to_csv(full_output_filename, header=False, mode='a', index=None, sep="\t")
                else:
                    df.to_csv(full_output_filename, header=True, index=None, sep="\t")
                    append = True


    def _process_chunk_for_test(self, chunk_split, process_method):

        all_processed_rows = []
        first_feature_string = chunk_split[0][1]
        feature_type = self._get_feature_type(feature_string=first_feature_string)
        feature_count = self._get_feature_count(feature_string=first_feature_string)

        columns = self._get_column_names(feature_count, feature_type, process_method)
        process_func = self._get_process_func(process_method)

        for row in chunk_split:
            try:
                id_job, values_in_string = row
                float_features = self._get_float_feature_values(values_in_string)[1:]

                #perform standardization
                proc_features = process_func(float_features,feature_type)

                index_of_max_feature = np.argmax(float_features)
                means = self._stats_features[feature_type]["mean"]
                diff = np.abs(float_features[index_of_max_feature] - means[index_of_max_feature])

                result_row = np.concatenate(([id_job], proc_features, [index_of_max_feature,diff]))
                all_processed_rows.append(result_row)

            except Exception as e:
                logging.error(str(e) + "in row: " + str(row))

        df = pd.DataFrame(all_processed_rows, columns=columns)
        df[["id_job", "max_feature_"+str(feature_type)+"_index"]] =df[["id_job", "max_feature_"+str(feature_type)+"_index"]].astype(int)
        return df

    def _standardize_features(self, float_features, feature_type):
        means = self._stats_features[feature_type]["mean"]
        stds = self._stats_features[feature_type]["std"]
        return (float_features - means) / stds

    @staticmethod
    def _check_path(path):
        """
        Check if exists folder. If no - create such folder.
        :param path: path to folder
        :return: None
        """
        os.system("if [ ! -d " + path + " ]; then mkdir -p " + path + "; fi")

    def _get_column_names(self, feature_count, feature_type, process_method):
        column_name = "feature_" + str(feature_type)
        if process_method == "standardization":
            column_name += "_stand_"
        else:
            raise NotImplementedError

        columns = [column_name + str(i) for i in range(feature_count)]
        return ["id_job"] + columns +["max_feature_"+str(feature_type)+"_index", "max_feature_"+str(feature_type)+"_abs_mean_diff"]

    def _get_process_func(self, process_method):
        if process_method == "standardization":
            return self._standardize_features
        else:
            raise NotImplementedError


if __name__ == "__main__":
    full_cmd_arguments = sys.argv
    argument_list = full_cmd_arguments[1:]

    arg_dict = {}
    if len(argument_list) > 0:
        long_options = ["train_filename=", "test_filename=", "output_folder=", "output_filename=", "process_method="]
        short_options = ""
        arguments, _ = getopt.getopt(argument_list, short_options, long_options)

        arguments = np.array(arguments)
        argument_keys = [arg_key[2:] for arg_key in arguments[:, 0]]
        arg_dict = dict(zip(argument_keys, arguments[:, 1]))


    train_arg={
        "train_filename": "data/train.tsv"
    }

    if arg_dict.get("train_filename", False):
        train_arg["train_filename"] = arg_dict["train_filename"]

    test_arg = {
        "test_filename": "data/test.tsv",
        "output_folder": "result",
        "output_filename": "test_proc.tsv",
        "process_method": "standardization"
    }

    for key in test_arg.keys():
        if arg_dict.get(key, False):
            test_arg[key] = arg_dict[key]

    preprocessor = Preprocessor()
    preprocessor.fit_train(**train_arg)
    preprocessor.process_test(**test_arg)
