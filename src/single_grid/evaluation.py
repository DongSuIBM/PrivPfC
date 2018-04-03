import sys, os, shutil
import operator
import pandas as pd

import single_grid
import tools
import analyse_single_grid

sys.path.append("..")
import utilities
import UG
import UG.UG
import config
import single_grid
import tree_ft
import tree_ft.tree_ft
from utilities.utilities import *
from utilities.common_classes import *


def batch_grid_evalution():
    
    print "---------------------------\n"
    print "--   Batch Grid Start   --\n"
    print "---------------------------\n"
    print "\n"
    
    grid_eval_dir = "grid_eval"
    if os.path.isdir(grid_eval_dir) == True:
        shutil.rmtree(grid_eval_dir)
    os.mkdir(grid_eval_dir)
    
    epsilon_1 = 0.4
    epsilon_2 = 0.6
    grid_size_threshold = 4
    
    dataset, config.config.taxonomy_forest, config.config.all_attributes = tools.loading_dataset()
    
    #analyse_grid_fast(dataset, epsilon_2)
    
    feasible_grids_list = single_grid.recursive_enumeration(grid_size_threshold)
    
    print "len(feasible_grids_list) = ", len(feasible_grids_list)
    
    #print feasible_grids_list
    
    single_grid.analyse_grid(feasible_grids_list, dataset, epsilon_1, epsilon_2, mode = "noise-free")
    
    #tree.predict()
    print "\n"
    print "--------------------------\n"
    print "--   Batch Grid End    --\n"
    print "--------------------------\n"
    print '-----All Set------------\n'
    
    


def customized_grid_evalution():
    
    print "---------------------------\n"
    print "--   Customized Grid Start   --\n"
    print "---------------------------\n"
    print "\n"
    
    grid_eval_dir = "grid_eval"
    if os.path.isdir(grid_eval_dir) == True:
        shutil.rmtree(grid_eval_dir)
    os.mkdir(grid_eval_dir)
    
    epsilon_1 = 0.4
    epsilon_2 = 0.6
    
    dataset, config.config.taxonomy_forest, config.config.all_attributes = tools.loading_dataset()
    
    #analyse_grid_fast(dataset, epsilon_2)
    
    selected_features = ["capital-gain", "capital-loss", "education", "marital-status", "relationship", "education-num"]
    
    feasible_grids_list = single_grid.customize_grid(selected_features)
    
    print "len(feasible_grids_list) = ", len(feasible_grids_list)
    
    #print feasible_grids_list
    
    single_grid.analyse_grid(feasible_grids_list, dataset, epsilon_1, epsilon_2, mode = "noise-free")
    
    #tree.predict()
    print "\n"
    print "--------------------------\n"
    print "--   Customized Grid End    --\n"
    print "--------------------------\n"
    print '-----All Set------------\n'

    
    
def single_grid_evaluation():
    
    print "---------------------------\n"
    print "--   Single Grid Start   --\n"
    print "---------------------------\n"
    print "\n"

    grid_eval_dir = "grid_eval"
    if os.path.isdir(grid_eval_dir) == True:
        shutil.rmtree(grid_eval_dir)
    os.mkdir(grid_eval_dir)

    grid_list_dir = "grid_list"
    if os.path.isdir(grid_list_dir) == True:
        shutil.rmtree(grid_list_dir)
    os.mkdir(grid_list_dir)

    epsilon_1 = 0.04
    epsilon_2 = 0.06
    
    dataset, config.config.taxonomy_forest, config.config.all_attributes = tools.loading_dataset()
    
    contin_tab = dataset.groupby(config.config.all_attributes + ["class"]).size()
    
    
    grid_0 = {
            "capital-gain": config.config.taxonomy_forest["capital-gain"].horizontal_hcs[2],
            "capital-loss": config.config.taxonomy_forest["capital-loss"].horizontal_hcs[2],
            "education": config.config.taxonomy_forest["education"].horizontal_hcs[2],
            "marital-status": config.config.taxonomy_forest["marital-status"].horizontal_hcs[1],
            "relationship": config.config.taxonomy_forest["relationship"].horizontal_hcs[1]
            }
    
    grid_1 = {
            "capital-gain": config.config.taxonomy_forest["capital-gain"].horizontal_hcs[2],
            "capital-loss": config.config.taxonomy_forest["capital-loss"].horizontal_hcs[2],
            "education": config.config.taxonomy_forest["education"].horizontal_hcs[2],
            "marital-status": config.config.taxonomy_forest["marital-status"].horizontal_hcs[3],
            "relationship": config.config.taxonomy_forest["relationship"].horizontal_hcs[1]
            }

    grid_2 = {
            "capital-gain": config.config.taxonomy_forest["capital-gain"].horizontal_hcs[2],
            "capital-loss": config.config.taxonomy_forest["capital-loss"].horizontal_hcs[2],
            "education": config.config.taxonomy_forest["education"].horizontal_hcs[2],
            "marital-status": config.config.taxonomy_forest["marital-status"].horizontal_hcs[3],
            "relationship": config.config.taxonomy_forest["relationship"].horizontal_hcs[2]
            }

    grid_3 = {
            "capital-gain": config.config.taxonomy_forest["capital-gain"].horizontal_hcs[2],
            "capital-loss": config.config.taxonomy_forest["capital-loss"].horizontal_hcs[2],
            "education": config.config.taxonomy_forest["education"].horizontal_hcs[3],
            "marital-status": config.config.taxonomy_forest["marital-status"].horizontal_hcs[3],
            "relationship": config.config.taxonomy_forest["relationship"].horizontal_hcs[2]
            }

    grid_4 = {
            "capital-gain": config.config.taxonomy_forest["capital-gain"].horizontal_hcs[2],
            "capital-loss": config.config.taxonomy_forest["capital-loss"].horizontal_hcs[2],
            "education": config.config.taxonomy_forest["education"].horizontal_hcs[4],
            "marital-status": config.config.taxonomy_forest["marital-status"].horizontal_hcs[3],
            "relationship": config.config.taxonomy_forest["relationship"].horizontal_hcs[2]
            }
    
    
    feasible_grids_list = [grid_0, grid_1, grid_2, grid_3, grid_4]
    
    single_grid.analyse_grid(feasible_grids_list, dataset, epsilon_1, epsilon_2, mode = "noisy")
    
    print "\n"
    print "--------------------------\n"
    print "--   Single Grid End    --\n"
    print "--------------------------\n"
    print '-----All Set------------\n'

if __name__ == "__main__":
    
    customized_grid_evalution()
    
    #batch_grid_evalution()
    
    #analyse_single_grid.analyse_one_grid_noisy(4)
    
    #single_grid_evaluation()
    
    print "Done!!!"
    