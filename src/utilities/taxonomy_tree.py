
import json
import random, copy, math
import numpy as np
import sys
import Queue
import config
import itertools

class TaxNode:
    def __init__(self, attribute, rep, isLeaf = False, level = 0):
        self.attribute = copy.deepcopy(attribute)
        self.key = copy.deepcopy(rep)
        self.children = []
        self.pseudo_child = None #only for generating
        self.is_leaf = isLeaf
        self.level = level
        self.hc_list = [] #the str list of hierarchy cut
        self.hc_node_list = []
        self.is_explored = False
        self.numerical_range = None #use tuple to represent the (a, b), [a, b]
        self.default_initial_bin_size = None
        self.selected_hc = None
        self.tree_depth = 0

        self.horizontal_hcs = {} #indexed by the corresponding level of the hcs.

        self.parent = None
        #Only valid for the root node,
        self.leaves = None

        #Only valid for the leaf node, just for the ease of search
        self.next_leaf = None

        self.oneR_accurate_count = 0

    def __gt__(self, other):
        return self.attribute > other.attribute

    def __lt__(self, other):
        return self.attribute < other.attribute

    def __ge__(self, other):
        return self.attribute >= other.attribute

    def __le__(self, other):
        return self.attribute <= other.attribute

    #if the current is root, get all the leaves
    def get_all_leaves(self):
        leaves_str_list = []

        p = self.leaves
        while p is not None:
            leaves_str_list.append(p.key)
            p = p.next_leaf

        return leaves_str_list

    #get all the leaves covered by this node
    def get_covered_leaves(self):
        leaves_str_list = []
        node_q = Queue.Queue()

        node_q.put(self)
        while not node_q.empty():
            curr = node_q.get()
            if curr.is_leaf == True:
                leaves_str_list.append(curr.key)
            for child in curr.children:
                node_q.put(child)

        return leaves_str_list

    def __str__(self):
        """
        result = '\t' * self.level + self.key
        if self.parent is not None:
            result += ', ' + self.parent.key
        return result
        """
        return self.key


    def __repr__(self):
        return self.__str__()

def construct_taxonomy_tree_from_json(tree_dict, attribute, level, leaves_list):

    child_node_list = []
    for value, subtree_dict in tree_dict.iteritems():
        if len(subtree_dict) == 0:
            node = TaxNode(attribute, value, True, level + 1)
            leaves_list.append(node)
        else:
            node = TaxNode(attribute, value, False, level + 1)
        node.children = construct_taxonomy_tree_from_json(subtree_dict, attribute, level + 1, leaves_list)
        child_node_list.append(node)

    return child_node_list

def construct_taxonomy_forest_from_json(dataset_name, numerical_attributes):
    forest_dict = None
    with open("../../datasets/crossvalidation/" + dataset_name + "/" + dataset_name + "_hierarchy.json", 'r') as f:
        forest_dict = json.load(f)

    taxonomy_forest = {}

    for attribute, tree in forest_dict.iteritems():
        #print "attribute = ", attribute
        leaves_list = []
        root_nodes = construct_taxonomy_tree_from_json(tree, attribute, -1, leaves_list)
        taxonomy_forest[attribute] = root_nodes[0]
        taxonomy_forest[attribute].leaves = leaves_list

        taxonomy_forest[attribute].tree_depth = max([leaf.level for leaf in leaves_list])

        if attribute in numerical_attributes:
            min_val = float(taxonomy_forest[attribute].key.split(",")[0][1:])
            max_val = float(taxonomy_forest[attribute].key.split(",")[1][1:-1])
            taxonomy_forest[attribute].numerical_range = tuple([min_val, max_val])
            taxonomy_forest[attribute].key = "Any"

        """
        print "================attribute = %s=================" % (attribute, )
        print_taxonomy_tree(taxonomy_forest[attribute])
        """

    return taxonomy_forest

def make_horizontal_hc(taxonomy_forest):

    taxonomy_forest_copy = copy.deepcopy(taxonomy_forest)

    for attribute, root in taxonomy_forest_copy.iteritems():
        #print "attribute = ", attribute
        for leaf in root.leaves:
            if leaf.level < root.tree_depth:
                for i in xrange(root.tree_depth - leaf.level):
                    pseudo_leaf = copy.deepcopy(leaf)
                    pseudo_leaf.level += 1
                    leaf.children = [pseudo_leaf]
                    leaf = leaf.children[0]
        #print_taxonomy_tree(taxonomy_forest_copy[attribute])

    for attribute, root in taxonomy_forest_copy.iteritems():

        #print attribute

        for i in xrange(root.tree_depth + 1):
            root.horizontal_hcs[i] = []

        node_q = Queue.Queue()
        node_q.put(root)
        while not node_q.empty():
            curr = node_q.get()
            root.horizontal_hcs[curr.level].append(curr.key)
            for child in curr.children:
                node_q.put(child)

    horizontal_hc_dict = {}
    for attribute, root in taxonomy_forest_copy.iteritems():
        horizontal_hc_dict[attribute] = root.horizontal_hcs

    return horizontal_hc_dict


#use BFS to search the given node in a tanonomy tree root
def search_node(root, attr_val):


    node_q = Queue.Queue()

    node_q.put(root)
    while not node_q.empty():
        curr = node_q.get()
        if curr.key == attr_val:
            return curr
        for child in curr.children:
            node_q.put(child)

    return None

#given a hierarchy cut and an actual attribute, find out the node in the hierarchy cut which cover the actual attribute

#hc is a list of string
def get_min_parent_node(taxonomy_tree_root, hc, attribute_str):

    return

#do the DPS and connect the child to its parent node. 
def DFS(curr):

    curr.is_explored = True

    for child in curr.children:
        if child.is_explored == False:
            DFS(child)
        child.parent = curr

def print_taxonomy_tree(taxRoot):

    #print 'print taxonomy tree for attribute %s' % (taxRoot.attribute)
    print taxRoot
    for child in taxRoot.children:
        print_taxonomy_tree(child)


def print_hierarchy_cut(hc_list):

    print 'print_hierarchy_cut:'
    output_buf = ''

    cnt = 0
    for hc in hc_list:
        output_buf = str(cnt) + ': [ '
        for elem in hc:
            output_buf += elem + ' '
        output_buf += ']'
        print output_buf
        cnt += 1


def generate_full_combinations(num_list):
    result = []
    for i in range(1, len(num_list)+1):
        result.append(list(itertools.combinations(num_list, i)))

    return result

def get_indexes_in_one_hc(hc):
    result = []
    for i in xrange(len(hc)):
        if hc[i].is_leaf == False:
            result.append(i)

    return result


def generate_all_hierarchy_cut_light_weight(taxRoot):
    hc_list = []
    hc_q = Queue.Queue()

    hc_q.put(taxRoot)
    non_leaf_cnt = 1
    i = 0

    while non_leaf_cnt != 0 and i < min(5, 2**taxRoot.tree_depth):
        hc = tuple(sorted([elem.key for elem in list(hc_q.queue)]))

        curr = hc_q.get()
        non_leaf_cnt -= 1

        if curr.is_leaf == True:
            hc_q.put(curr)
            #non_leaf_cnt -= 1
        else:
            for child in curr.children:
                hc_q.put(child)
                non_leaf_cnt += 1

        hc_list.append(hc)
#		print 'hc_list = ', hc_list
        i += 1

    hc_set = set(hc_list)

#	print "hc_set = ", hc_set

    return list(hc_set)

def generate_all_hierarchy_cut(taxRoot):

    hc_list_1 = []
    hc_q = Queue.Queue()

    print_taxonomy_tree(taxRoot)

    hc_q.put([taxRoot])
    while not hc_q.empty():
        curr = hc_q.get()

        hc_list_1.append(copy.deepcopy(curr))
        splitable_idx = get_indexes_in_one_hc(curr)
        if len(splitable_idx) == 0:
            break

        comb = generate_full_combinations(splitable_idx)
        for c_strategy in comb:
            for t in c_strategy:
                for elem in t:
                    new_hc = copy.deepcopy(curr)
                    if curr[elem].is_leaf == False:
                        new_hc = new_hc + curr[elem].children
                        new_hc.pop(elem)
                        hc_q.put(new_hc)

    #The purpose here to use string is to remove duplicate hc
    hc_str_tuple_list = []
    for hc in sorted(hc_list_1):
        hc_str = []
        for elem in hc:
            hc_str.append(elem.key.strip())
        hc_str.sort()
        hc_str_tuple_list.append(tuple(hc_str))

    hc_list_tobe_return = list(set(hc_str_tuple_list))
    hc_list_tobe_return.sort(key = lambda s: len(s))
    taxRoot.hc_list = copy.deepcopy(hc_list_tobe_return)

#	print "taxRoot.hc_list = ", taxRoot.hc_list

    return hc_list_tobe_return

