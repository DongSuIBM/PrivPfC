import sys
import pandas as pd
import copy
import os.path

sys.path.append("..")

import config
import config.config
import utilities
import utilities.taxonomy_tree


def loading_adult_dataset_crossvalidation(fold_num):
    dataset_name = "adult"
    dataset = pd.read_csv("../../datasets/crossvalidation/" + dataset_name + "/folds/" + dataset_name + "_" +
                          str(fold_num) + "_train.txt")

    all_attributes = ["age","workclass","fnlwgt","education","education-num",\
                "marital-status","occupation","relationship","race","sex",
                "capital-gain","capital-loss","hours-per-week","native-country"]

    categorical_attributes = [u'workclass', u'education', u'marital-status', u'occupation', u'relationship', u'race',
                              u'sex', u'native-country']
    numerical_attributes = ['age', 'capital-gain', 'capital-loss', 'education-num', 'hours-per-week', 'fnlwgt']
    taxonomy_forest = utilities.taxonomy_tree.construct_taxonomy_forest_from_json(dataset_name, numerical_attributes)
    horizontal_hc_dict =  utilities.taxonomy_tree.make_horizontal_hc(taxonomy_forest)
    class_info = {"class": ["<=50K", ">50K"]}

    config.config.dataset_info["dataset_name"] = dataset_name
    config.config.dataset_info["taxonomy_forest"] = taxonomy_forest
    config.config.dataset_info["horizontal_hc_dict"] = horizontal_hc_dict
    config.config.dataset_info["all_attributes"] = all_attributes
    config.config.dataset_info["selected_attributes"] = all_attributes
    config.config.dataset_info["class_info"] = class_info
    config.config.dataset_info["categorical_attributes"] = categorical_attributes
    config.config.dataset_info["numerical_attributes"] = numerical_attributes
    config.config.dataset_info["all_attributes_extended"] = all_attributes + ["class"]
    config.config.dataset_info["I_tab_key_sep"] = "_"

    bucketized_fn = "../../datasets/crossvalidation/" + dataset_name + "/folds/" + dataset_name + "_" + \
                    str(fold_num) + "_train.bucketized" + ".txt"

    if os.path.isfile(bucketized_fn) == False:
        dataset = bucketize_numerical_attributes(copy.deepcopy(dataset), numerical_attributes,
                                                 config.config.dataset_info["all_attributes_extended"])
        dataset.to_csv(bucketized_fn, index=False)
    else:
        dataset = pd.read_csv(bucketized_fn)

    config.config.dataset_info["orig_dataset"] = dataset

    print "loading_adult_dataset done"

    return

"""
Load the adult dataset with marital-status as the class
"""
def loading_adult_marital_status_dataset_crossvalidation(fold_num):
    dataset_name = "adult_marital-status"
    dataset = pd.read_csv("../../datasets/crossvalidation/" + dataset_name + "/folds/" + dataset_name + "_"
                          + str(fold_num) + "_train.txt")

    all_attributes = ["age","workclass","fnlwgt","education","education-num",\
                "income","occupation","relationship","race","sex",
                "capital-gain","capital-loss","hours-per-week","native-country"]

    numerical_attributes = ['age', 'capital-gain', 'capital-loss', 'education-num', 'hours-per-week', 'fnlwgt']
    taxonomy_forest = utilities.taxonomy_tree.construct_taxonomy_forest_from_json(dataset_name, numerical_attributes)
    horizontal_hc_dict =  utilities.taxonomy_tree.make_horizontal_hc(taxonomy_forest)
    class_info = {"class": ["Married-Partner-present", "Married-Partner-absent", "Never-married"]}

    config.config.dataset_info["dataset_name"] = dataset_name
    config.config.dataset_info["taxonomy_forest"] = taxonomy_forest
    config.config.dataset_info["horizontal_hc_dict"] = horizontal_hc_dict
    config.config.dataset_info["all_attributes"] = all_attributes
    config.config.dataset_info["selected_attributes"] = all_attributes
    config.config.dataset_info["class_info"] = class_info
    config.config.dataset_info["numerical_attributes"] = numerical_attributes
    config.config.dataset_info["all_attributes_extended"] = all_attributes + ["class"]
    config.config.dataset_info["I_tab_key_sep"] = "_"

    bucketized_fn = "../../datasets/crossvalidation/" + dataset_name + "/folds/" + dataset_name + "_"\
                    + str(fold_num) + "_train.bucketized" + ".txt"
    if os.path.isfile(bucketized_fn) == False:
        dataset = bucketize_numerical_attributes(copy.deepcopy(dataset), numerical_attributes,
                                                 config.config.dataset_info["all_attributes_extended"])
        dataset.to_csv(bucketized_fn, index=False)
    else:
        dataset = pd.read_csv(bucketized_fn)

    config.config.dataset_info["orig_dataset"] = dataset

    print "loading_adult_marital-status_dataset done"

    return

def bin_lookup(bins_fact, idx):
    bins_label = bins_fact.labels
    bins = bins_fact.levels
    return bins[bins_label[idx]]

def bucketize_numerical_attributes(dataset, numerical_attributes, schema):

    gen_list_dict = {}
    for attr in numerical_attributes:
        tree = config.config.dataset_info["taxonomy_forest"][attr]
        cutting_points = []
        for leaf in tree.leaves:
            min_val = float(leaf.key.split(",")[0][1:])
            max_val = float(leaf.key.split(",")[1][1:-1])
            cutting_points.append(min_val)
            cutting_points.append(max_val)
        cutting_points = sorted(list(set(cutting_points)))
        bins_fact = pd.cut(dataset[attr], cutting_points)
        gen_list_dict[attr] = [bins_fact.values[i] for i in xrange(len(bins_fact.values))]

    for n_attr in numerical_attributes:
        del dataset[n_attr]

    for n_attr in numerical_attributes:
        dataset[n_attr] = pd.Series(gen_list_dict[n_attr])

    dataset = dataset.reindex(columns=schema)
    return dataset


