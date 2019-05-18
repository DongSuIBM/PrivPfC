import sys
import itertools
import math
import os
import pandas as pd
import numpy as np
import json, copy
import multiprocessing
import time

import tools

sys.path.append("..")
import config
import utilities

def cell_quality_binary_classification(N_pos, N_neg, epsilon_2):
    max_term = max(N_pos, N_neg)
    min_term = min(N_pos, N_neg)
    pr_max = 0.5 * math.e ** (-1.0 * epsilon_2 * abs(N_pos - N_neg)) * (1.0 + 0.5 * epsilon_2 * abs(N_pos - N_neg))
    pr_min = 1.0 - pr_max
    score = max_term * pr_max + min_term * pr_min
    return score

def cell_quality_multiclass_classification(class_count_s, epsilon_PT):
    class_count_sorted_dec = sorted(class_count_s.values.tolist(), reverse=True)
    top_1_2 = class_count_sorted_dec[:2]
    score = cell_quality_binary_classification(top_1_2[0], top_1_2[1], epsilon_PT)
    return score

"""
The worse case constant upper bound for the quality function sensitivity, which 1.1 as shown in Section 4.2
"""
def quality_function_binary_sensitivity():
    return 1.1

def quality_function_multi_sensitivity():
    return 1.1

"""
Only used by the exponential mechanism.
"""
def grid_quality_binary(I_tab, grid, epsilon_2):

    quality_bi_cls_total = 0.0

    cell_size_list = []

    for key, item in I_tab.iteritems():

        N_pos = item[config.config.dataset_info["class_info"]["class"][0]]
        N_neg = item[config.config.dataset_info["class_info"]["class"][1]]
        N_cell_samples = N_pos + N_neg
        cell_size_list.append(N_cell_samples)

        quality_bi_cls = cell_quality_binary_classification(N_pos, N_neg, epsilon_2)

        quality_bi_cls_total += quality_bi_cls

    results = {}

    results["grid"] = grid

    results['stats'] = {"quality_binary_classification": quality_bi_cls_total}

    return results


def grid_quality_multi(I_tab, grid, epsilon_2):

    quality_multi_cls_total = 0.0
    cell_size_list = []
    total_class_count_s = pd.Series(0, index=config.config.dataset_info["class_info"]["class"])

    i = 0
    for key, item in sorted(I_tab.iteritems()):
        class_count_s = pd.Series(item)

        total_class_count_s += class_count_s
        N_cell_samples = class_count_s.sum()
        cell_size_list.append(N_cell_samples)

        quality_7 = cell_quality_multiclass_classification(class_count_s, epsilon_2)

        quality_multi_cls_total += quality_7

        i += 1

    results = {}

    results["grid"] = grid

    results['stats'] = {"quality_multiclass_classification": quality_multi_cls_total}

    return results



def dump_train_test(I_tab, grid):
    training_data = synthetically_generate(I_tab, grid)
    testing_data = generalize_test_dataset(config.config.dataset_info["dataset_name"], grid,
                                           config.config.dataset_info["all_attributes_extended"],
                                           config.config.dataset_info["numerical_attributes"])

    dir_str = config.config.root_dir + "_".join(["eps=" + str(config.config.exp_info["epsilon"]),
                                                "fold=" + str(config.config.exp_info["fold_num"])])

    training_data.to_csv(dir_str + "_" + "train.csv", index=False)
    testing_data.to_csv(dir_str + "_" + "test.csv", index=False)
    return

def drill_down(grid, low_level_keys):

    hier_mapping = {}
    mapping = {}

    selected_attrs = sorted(grid.keys())
    extended_selected_attrs = selected_attrs + ["class"]
    for attr, vals in grid.items():

        hier_mapping[attr] = {}
        hier_mapping["class"] = {}
        for val in vals:
            node = utilities.taxonomy_tree.search_node(config.config.dataset_info["taxonomy_forest"][attr], val)
            leaves = node.get_covered_leaves()
            for leaf in leaves:
                hier_mapping[attr][leaf] = val

        for i in xrange(len(config.config.dataset_info["class_info"]["class"])):
            hier_mapping["class"][config.config.dataset_info["class_info"]["class"][i]] = config.config.dataset_info["class_info"]["class"][i]

    #convert local attribute index to the index in all attribute list
    l2g = {}
    extended_all_attributes = config.config.dataset_info["all_attributes_extended"]
    for i in xrange(len(extended_selected_attrs)):
        l2g[i] = extended_all_attributes.index(extended_selected_attrs[i])

    for key in low_level_keys:
        value = []
        for idx in xrange(len(extended_selected_attrs)):
            attr = extended_selected_attrs[idx]
            leaf = key[l2g[idx]]
            value.append(hier_mapping[attr][leaf])

        mapping[key] = tuple(value)

    return mapping


def reload_grid_list():
    grid_list_file_path = "_".join(["eps=" + str(config.config.exp_info["epsilon"]),
                                    "fold=" + str(config.config.exp_info["fold_num"]), "grid_list.json"])

    with open(config.config.root_dir + grid_list_file_path, 'r') as f:
        grid_list = json.load(f)
    return grid_list

def dump_selected_grid(grid):
    grid_fn = config.config.root_dir + "_".join(["eps=" + str(config.config.exp_info["epsilon"]),
                                                "fold=" + str(config.config.exp_info["fold_num"]), "selected_grid.json"])

    with open(grid_fn, "w") as f:
        json.dump(grid, f)

    return


def gen_1_attr_level_itemsets():
    meta_grid_list = []

    selected_attributes = sorted(config.config.dataset_info["selected_attributes"])
    for attr in selected_attributes:
        hc_list = config.config.dataset_info["horizontal_hc_dict"][attr]
        for level in hc_list.keys():
            if level == 0:
                continue
            grid_size = len(hc_list[level])
            meta_grid = {"grid": {attr: level}, "join_key": "", "attr_list": [attr], "size":grid_size}
            meta_grid_list.append(meta_grid)

    return meta_grid_list

def join_meta_grids(meta_grid_1, meta_grid_2):
    joined = copy.deepcopy(meta_grid_1)
    inc_attr = joined['attr_list'][-1]
    if len(joined["join_key"]) == 0:
        joined["join_key"] = "_".join([inc_attr, str(joined['grid'][inc_attr])])
    else:
        joined["join_key"] = joined["join_key"] + "~" + "_".join([inc_attr, str(joined['grid'][inc_attr])])

    new_attr = copy.deepcopy(meta_grid_2['attr_list'][-1])
    joined['grid'][new_attr] = meta_grid_2["grid"][new_attr]
    joined['attr_list'].append(new_attr)
    joined['size'] = meta_grid_1['size'] * len(config.config.dataset_info["horizontal_hc_dict"][new_attr][joined['grid'][new_attr]])
    return joined

def test_join_condition(meta_grid_1, meta_grid_2, grid_size_threshold):

    if meta_grid_1["join_key"] != meta_grid_2["join_key"] or meta_grid_1["attr_list"][-1] == meta_grid_2["attr_list"][-1]:
        result = False
    else:
        new_attr = meta_grid_2["attr_list"][-1]
        size = meta_grid_1["size"] * len(config.config.dataset_info["horizontal_hc_dict"][new_attr][meta_grid_2['grid'][new_attr]])
        if size > grid_size_threshold:
            result = False
        else:
            result = True

    return result

def candidate_grid_generation(prev_meta_grid_list, total_num_grids, Omega, grid_size_threshold):
    max_list_len = Omega
    meta_grid_list_new_tmp = [None for i in xrange(max_list_len)]
    curr_num_grids = 0

    for mgid_1 in xrange(len(prev_meta_grid_list)):

        for mgid_2 in xrange(mgid_1 + 1, len(prev_meta_grid_list)):

            m_grid_1 = prev_meta_grid_list[mgid_1]
            m_grid_2 = prev_meta_grid_list[mgid_2]

            cond = test_join_condition(m_grid_1, m_grid_2, grid_size_threshold)

            if cond == True:
                joined = join_meta_grids(m_grid_1, m_grid_2)
                meta_grid_list_new_tmp[curr_num_grids] = joined
                curr_num_grids += 1

                if total_num_grids + curr_num_grids >= Omega:
                    print "Reaching the pool size limit, done"
                    meta_grid_list_new = meta_grid_list_new_tmp[:curr_num_grids]
                    return meta_grid_list_new, curr_num_grids

    meta_grid_list_new = meta_grid_list_new_tmp[:curr_num_grids]

    return meta_grid_list_new, curr_num_grids

def iterative_enumeration(Omega, grid_size_threshold):

    print "iterative_enumeration, Omega = %d, T = %d" % (Omega, grid_size_threshold)

    meta_grid_list_dict = {}

    meta_1_grid_list = gen_1_attr_level_itemsets()

    meta_grid_list_dict[1] = meta_1_grid_list

    print "1, %d" % (len(meta_1_grid_list), )

    total_num_grids = len(meta_1_grid_list)

    for it in xrange(2, len(config.config.dataset_info["selected_attributes"])+1):

        prev_meta_grid_list = meta_grid_list_dict[it-1]
        meta_grid_list_new, curr_num_grids = candidate_grid_generation(prev_meta_grid_list, total_num_grids, Omega, grid_size_threshold)
        meta_grid_list_dict[it] = copy.deepcopy(meta_grid_list_new)
        total_num_grids += curr_num_grids
        if total_num_grids >= Omega:
            break

    max_it = it
    print "max_it = ", max_it

    grid_list = []
    for it in xrange(1, max_it+1):
        for mgid in xrange(len(meta_grid_list_dict[it])):
            meta_grid = meta_grid_list_dict[it][mgid]["grid"]
            grid = {}
            for attr in meta_grid.keys():
                grid[attr] = config.config.dataset_info["horizontal_hc_dict"][attr][meta_grid[attr]]
            grid_list.append(grid)

    grid_list_len = len(grid_list)
    print "len(grid_list) = ", grid_list_len

    config_str = "_".join(["eps="+str(config.config.exp_info["epsilon"]), "fold=" + str(config.config.exp_info["fold_num"]), "grid_list.json"])
    grid_list_store_path = config.config.root_dir + config_str
    with open(grid_list_store_path, 'w') as f:
        json.dump(grid_list, f)

    return grid_list_len

def construct_I_tab(grid, contin_tab):

    I_tab = {}
    empty_cell = {}
    for class_label in config.config.dataset_info["class_info"]["class"]:
        empty_cell[class_label] = 0

    hc_list = [grid[key] for key in sorted(grid.keys())]

    for I_tab_key in [key for key in itertools.product(*hc_list)]:
        new_I_tab_key = config.config.dataset_info["I_tab_key_sep"].join(I_tab_key)
        I_tab[new_I_tab_key] = copy.deepcopy(empty_cell)

    grid_mapping = drill_down(grid, contin_tab.index.tolist())

    for key, value in contin_tab.iteritems():
        gen_key = grid_mapping[key]
        I_tab_key = config.config.dataset_info["I_tab_key_sep"].join(gen_key[:-1])
        class_key = gen_key[-1]
        I_tab[I_tab_key][class_key] += value

    return I_tab

def do_parallel_compute_quality_score_list(grid_list, epsilon_2, start_gid, end_gid):
    quality_score_list = []

    common_prefix = config.config.root_dir + "/" + "_".join(["quality_list", "eps=" + str(config.config.exp_info["epsilon"]),
                                                             "fold=" + str(config.config.exp_info["fold_num"])]) + "/"

    if not os.path.exists(common_prefix):
        os.system('mkdir -p ' + common_prefix)

    for gid in xrange(start_gid, end_gid):
        grid = grid_list[gid]

        if gid % 100 == 0:
            print "start_gid = %d, end_gid = %d, gid = %d" % (start_gid, end_gid, gid)

        I_tab = construct_I_tab(grid, config.config.dataset_info["contin_tab"])
        quality_results = config.config.exp_info['quality_function'](I_tab, grid, epsilon_2)
        quality_score_list.append(quality_results['stats'][config.config.exp_info["quality_name"]])

    with open(common_prefix + 'quality_list_' + str(start_gid) + "-" + str(end_gid) + '.json', 'w') as f:
        json.dump(quality_score_list, f)

    return


def parallel_compute_quality_score_list(epsilon_2, step):

    grid_list = reload_grid_list()
    grid_list_size = len(grid_list)
    print "grid_list_size = ", grid_list_size, 
    pool = multiprocessing.Pool()
    for start_gid in xrange(0, grid_list_size, step):
        end_gid = min(start_gid + step, grid_list_size)
        print "start_gid = ", start_gid, ", end_gid = ", end_gid
        pool.apply_async(do_parallel_compute_quality_score_list, (grid_list, epsilon_2, start_gid, end_gid))

    pool.close()
    pool.join()
    #sleep 5s to wait for io
    time.sleep(5) 

    return

def select_grid(quality_score_list, epsilon_1):

    r = np.random.random()
    exponent_s = pd.Series(quality_score_list) * \
                 (-1.0 * epsilon_1 / (2.0 * config.config.exp_info["quality_sensitivity"]))

    max_exponent = max(exponent_s)
    exponent_shifted_s = exponent_s - max_exponent
    expon_func = lambda x : math.e ** x

    weight_array = pd.Series(exponent_shifted_s).map(expon_func).values
    p_array = weight_array * 1.0 / weight_array.sum()
    p_cum_array = p_array.cumsum()
    selected_idx = p_cum_array.searchsorted(r)

    return selected_idx

def prepare_data(dataset_name, fold_num):

    if dataset_name == "adult":
        tools.loading_adult_dataset_crossvalidation(fold_num)
    elif dataset_name == "adult_marital-status":
        tools.loading_adult_marital_status_dataset_crossvalidation(fold_num)
    else:
        print "Dataset not available"

    return


def reload_quality_score_list():
    common_prefix = config.config.root_dir + "/" + "_".join(["quality_list", "eps=" + str(config.config.exp_info["epsilon"]),
                                                             "fold=" + str(config.config.exp_info["fold_num"])]) + "/"

    quality_score_list = []
    for start_gid in xrange(0, config.config.grid_pool_size, config.config.quality_list_step):
        end_gid = min(start_gid + config.config.quality_list_step, config.config.grid_pool_size)
        with open(common_prefix + 'quality_list_' + str(start_gid) + "-" + str(end_gid) + '.json', 'r') as f:
            q_score_list = json.load(f)

        quality_score_list += q_score_list

    return quality_score_list

def selectHist(epsilon_PS):
    quality_score_list = reload_quality_score_list()
    selected_gid = select_grid(quality_score_list, epsilon_PS)
    grid_list = reload_grid_list()
    dump_selected_grid(grid_list[selected_gid])
#    print grid_list[selected_gid]
    return selected_gid


def perturbHist(selected_gid, epsilon_PT):
    grid_list = reload_grid_list()
    selected_grid = grid_list[selected_gid]
    selected_I_tab = construct_I_tab(selected_grid, config.config.dataset_info["contin_tab"])
    noisy_selected_I_tab = inject_laplace_noise(selected_I_tab, epsilon_PT)
    dump_train_test(noisy_selected_I_tab, selected_grid)
    return

def inject_laplace_noise(I_tab, epsilon_2):
    noisy_I_tab = copy.deepcopy(I_tab)
    for key in noisy_I_tab.keys():
        for cls_label in config.config.dataset_info["class_info"]["class"]:
            noisy_I_tab[key][cls_label] = noisy_I_tab[key][cls_label] + np.random.laplace(scale=1.0/epsilon_2)

    return noisy_I_tab


def do_generalization_according_to_grid(grid, dataset, taxonomy_forest, schema):

    for attr, hc in grid.items():
        mapping = {}
        for val in hc:
            node = utilities.taxonomy_tree.search_node(taxonomy_forest[attr], val)
            leaves = node.get_covered_leaves()
            for leaf in leaves:
                mapping[leaf] = val

        dataset[attr] = dataset[attr].map(mapping)

    remaining_attributes = set(schema).difference(set(grid.keys() + ["class"]))
    for attr in remaining_attributes:
        dataset[attr] = "Any"

    return

def generalize_test_dataset(dataset_name, grid, schema, numerical_attributes):

    test_fn = "../../datasets/crossvalidation/" + dataset_name + "/folds/" + dataset_name + "_" + \
              str(config.config.exp_info["fold_num"]) + "_test.txt"
    if config.config.dataset_info["dtype_dict"] is None:
        test_dataset = pd.read_csv(test_fn)
    else:
        test_dataset = pd.read_csv(test_fn, dtype=config.config.dataset_info["dtype_dict"])

    if len(numerical_attributes) > 0:
        test_dataset_gen = tools.bucketize_numerical_attributes(copy.deepcopy(test_dataset),
                                                                numerical_attributes,
                                                                config.config.dataset_info["all_attributes_extended"])
        test_dataset_gen = test_dataset_gen.reindex(columns=schema)
    else:
        test_dataset_gen = test_dataset

    do_generalization_according_to_grid(grid, test_dataset_gen, config.config.dataset_info["taxonomy_forest"], schema)

    return test_dataset_gen

def do_attribute_selection_EM_selection(attr_name_score_list, working_epsilon, sensitivity):

    attr_score_list = [elem[1] for elem in attr_name_score_list]

    r = np.random.random()
    exponent_s = pd.Series(attr_score_list) * (working_epsilon / (2.0 * sensitivity))
    max_exponent = max(exponent_s)
    exponent_shifted_s = max_exponent - exponent_s
    expon_func = lambda x: math.e ** (-1.0 * x)

    weight_array = pd.Series(exponent_shifted_s).map(expon_func).values
    p_array = weight_array * 1.0 / weight_array.sum()
    p_cum_array = p_array.cumsum()
    selected_idx = p_cum_array.searchsorted(r)

    return selected_idx


def estimate_grid_size_threshold(N, PT_epsilon):

    """
    |G|/(epsilon_PT) <= N * Delta
    where Delta = 0.2 or depend on the dataset
    """
    threshold = int(round(N * PT_epsilon * 1.0 * config.config.ND_delta))
    return threshold

def estimate_N(N, epsilon_N):
    N_hat = N + np.random.laplace(scale=1.0/epsilon_N)
    return N_hat

def PrivateHistogramPublishing(epsilon_PS, epsilon_PT):
    parallel_compute_quality_score_list(epsilon_PT, config.config.quality_list_step)
    selected_gid = selectHist(epsilon_PS)
    print "selected_gid = ", selected_gid
    perturbHist(selected_gid, epsilon_PT)
    return

def privpfc_main_routine(epsilon, Omega):

    print "privpfc_main_routine"
    epsilon_N = epsilon * config.config.exp_info["N_ratio"]
    epsilon_PS = epsilon * config.config.exp_info["PS_ratio"]
    epsilon_PT = epsilon * config.config.exp_info["PT_ratio"]

    print "\t".join(["epsilon_N", str(epsilon_N), "epsilon_PS", str(epsilon_PS), "epsilon_PT", str(epsilon_PT)]) + "\n"

    config.config.dataset_info["dataset"] = config.config.dataset_info["orig_dataset"]
    config.config.dataset_info["contin_tab"] = \
        config.config.dataset_info["dataset"].groupby(config.config.dataset_info["all_attributes_extended"]).size()

    N = config.config.dataset_info["dataset"].shape[0]
    N_hat = estimate_N(N, epsilon_N)

    T = estimate_grid_size_threshold(N_hat, epsilon_PT)
    config.config.grid_pool_size = iterative_enumeration(Omega, T)
    config.config.quality_list_step = int(math.ceil(config.config.grid_pool_size / config.config.num_available_cores))

    print "\t".join(["N", str(N), "N_hat", str(N_hat), "T", str(T),
                     "grid_pool_size", str(config.config.grid_pool_size),
                     "quality_list_step", str(config.config.quality_list_step)]) + "\n"

    PrivateHistogramPublishing(epsilon_PS, epsilon_PT)

    return


def synthetically_generate(I_tab, grid):

    synthe_data_list = []

    selected_attrs = sorted(grid.keys())
    schema = config.config.dataset_info["all_attributes_extended"]

    count_dict = {}
    for cls_label in config.config.dataset_info["class_info"]["class"]:
        count_dict[cls_label] = 0

    for key, cell_stat in I_tab.iteritems():

        listed_key = key.split(config.config.dataset_info["I_tab_key_sep"])
        synthe_data = {}
        for attr in config.config.dataset_info["all_attributes"]:
            if grid.has_key(attr) == True:
                synthe_data[attr] = listed_key[selected_attrs.index(attr)]
            else:
                synthe_data[attr] = "Any"

        for cls_label in config.config.dataset_info["class_info"]["class"]:
            if round(cell_stat[cls_label]) > 0:
                synthe_data_copy = copy.deepcopy(synthe_data)
                synthe_data_copy["class"] = cls_label
                synthe_data_list += [synthe_data_copy] * int(round(cell_stat[cls_label]))
                count_dict[cls_label] += int(round(cell_stat[cls_label]))

    synthe_data_df = pd.DataFrame(synthe_data_list)
    synthe_data_df = synthe_data_df.reindex(columns=schema)

    return synthe_data_df
