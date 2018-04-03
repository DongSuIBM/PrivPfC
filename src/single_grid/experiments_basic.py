import sys
import os
import json
import single_grid
import config
import random
import numpy as np
sys.path.append("..")


#dataset_name, epsilon, fold_num, pool_size_threshold
def main(args):

    print "---------------------------\n"
    print "--   Experiment Start    --\n"
    print "---------------------------\n"
    print "\n"

    #set up the path to the results and all intermediate files
    config.config.root_dir = args['save'] + "/" + args['dataset_name'] + "/"
    if not os.path.exists(config.config.root_dir):
        os.system('mkdir -p ' + config.config.root_dir)

    with open("budget_allocation_plan.json") as f:
        budget_allocation_plan = json.load(f)

    budget_alloc_plan_key = "pfc"

    if args['dataset_name'] == 'adult':
        config.config.exp_info["quality_name"] = "quality_binary_classification"
        config.config.exp_info['quality_function'] = single_grid.grid_quality_binary
        config.config.exp_info["quality_sensitivity"] = single_grid.quality_function_binary_sensitivity()
    elif args['dataset_name'] == "adult_marital-status":
        config.config.exp_info["quality_name"] = "quality_multiclass_classification"
        config.config.exp_info['quality_function'] = single_grid.grid_quality_multi
        config.config.exp_info["quality_sensitivity"] = single_grid.quality_function_multi_sensitivity()
    else:
        print "invalid dataset name, exit"
        return

    #load privacy budget allocation to the experiment configuration structure
    config.config.exp_info["N_ratio"] = budget_allocation_plan[budget_alloc_plan_key]["N_ratio"]
    config.config.exp_info["PS_ratio"] = budget_allocation_plan[budget_alloc_plan_key]["PS_ratio"]
    config.config.exp_info["PT_ratio"] = budget_allocation_plan[budget_alloc_plan_key]["PT_ratio"]

    #Set the number of threads to evaluate the grid qualities in parallel, default value is 1
    config.config.num_available_cores = args['num_threads']

    #The noise-to-signal ratio, see Section 4.4 in the PrivPfC paper.
    config.config.ND_delta = args['noise_to_signal_ratio']

    #Total privacy budget
    config.config.exp_info["epsilon"] = args['epsilon']

    #The fold number of the train/test pair in the cross validation set
    config.config.exp_info["fold_num"] = args['fold_num']
    print "\t".join(["pool_size_threshold", str(args['pool_size_threshold']), "epsilon", str(args['epsilon']),
                     "fold_num", str(args['fold_num'])])

    single_grid.prepare_data(args['dataset_name'], args['fold_num'])
    single_grid.privpfc_main_routine(args['epsilon'], args['pool_size_threshold'])

    print "\n"
    print "--------------------------\n"
    print "--   Experiment End     --\n"
    print "--------------------------\n"
    print '-----All Set------------\n'


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_name", choices=["adult", "adult_marital-status"], default="adult")

    parser.add_argument("-s", "--save", default="../exp_data/")

    parser.add_argument("-e", "--epsilon", type=float, default=1.0)

    parser.add_argument("-f", "--fold_num", type=int, default=0)

    parser.add_argument("-p", "--pool_size_threshold", type=int, default=10000)

    parser.add_argument("-t", "--num_threads", type=int, default=1)

    parser.add_argument("-n", "--noise_to_signal_ratio", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=1215)

    args = vars(parser.parse_args())

    # setup random seed
    random.seed(args['seed'])
    np.random.seed(args['seed'])

    print(args)

    main(args)
