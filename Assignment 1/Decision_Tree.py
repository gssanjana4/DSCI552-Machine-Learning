"""
This Program is implemented as part of Assignment 1 for DSCI 552
Contributors: Sanjana Gopnal Swamy & Amit Sankhla
"""

import re
import math


class TreeNode:
    def __init__(self, name, index):
        self.node_name = name
        self.node_children = dict()
        self.node_index = index


class DecisionTree:
    features = []
    data = []
    root = None

    def get_data(self, path):
        feature_pattern = "\\((.*)\\)"
        data_pattern = "[0-9]+: (.*);"
        file = open(path, "r")
        for line in file:
            if re.search(feature_pattern, line):
                self.features = re.split(feature_pattern, line)[1].split(", ")
            else:
                if re.search(data_pattern, line):
                    self.data.append(re.split(data_pattern, line)[1].split(", "))

    def get_splitted_data(self, data, index):
        if data is None or len(data) == 0:
            return None
        res = dict()
        for row in data:
            key = row[index]
            if key not in res:
                res[key] = []
            res[key].append(row)

        return res

    def calculate_entropy(self, data):
        if data is None or len(data) == 0:
            return 0
        res = dict()
        index = len(self.features) - 1
        for row in data:
            res[row[index]] = res.get(row[index], 0) + 1

        ans = 0
        for value in res.values():
            p = value / len(data)
            ans += (-p * math.log(p))
        return ans

    def get_majority_label(self, data):
        if data is None or len(data) == 0:
            return None
        res = dict()
        index = len(self.features) - 1
        for row in data:
            res[row[index]] = res.get(row[index], 0) + 1
        return max(res, key=res.get)  # returns the key with max value

    def choose_best_feature(self, data, feature_indexes):
        base_entropy = self.calculate_entropy(data)
        best_information_gain = -1
        best_feature_index = -1
        for index in feature_indexes:
            splitted_data = self.get_splitted_data(data, index)
            current_entropy = 0
            for key in splitted_data.keys():
                p = len(splitted_data[key]) / len(data)
                current_entropy += p * self.calculate_entropy(splitted_data[key])

            current_information_gain = base_entropy - current_entropy
            if current_information_gain > best_information_gain:
                best_feature_index = index
                best_information_gain = current_information_gain

        return best_feature_index

    def build_decision_tree(self):
        feature_indexes = list(range(0, len(self.features) - 1))
        self.root = self.build_decision_tree_helper_function(self.data, feature_indexes)

    def build_decision_tree_helper_function(self, data, feature_indexes):
        if data is None or len(data) == 0:
            return None
        if feature_indexes is None or len(feature_indexes) == 0 or self.calculate_entropy(data) == 0:
            return TreeNode(self.get_majority_label(data), -1)  # -1 index for leaf node
        best_index = self.choose_best_feature(data, feature_indexes)
        if best_index < 0:
            return TreeNode(self.get_majority_label(data), -1)

        feature_indexes.remove(best_index)
        root = TreeNode(self.features[best_index], best_index)
        splitted_data = self.get_splitted_data(data, best_index)

        for key, value in splitted_data.items():
            child = self.build_decision_tree_helper_function(value, feature_indexes.copy())
            if child is not None:
                root.node_children[key] = child

        return root

    def display_tree(self):
        self.display_tree_helper_function(self.root, 1)

    def display_tree_helper_function(self, root, level):
        if root is None:
            return
        if root == self.root:
            print(root.node_name, end=" ")
        else:
            print(": ", root.node_name, end=" ")

        for key, value in root.node_children.items():
            print(" ")
            print(" " * level*5, end=" ")
            print(key, end=" ")
            self.display_tree_helper_function(value, level + 1)

    def predict_label(self, test_data):
        return self.predict_label_helper_function(self.root, test_data)

    def predict_label_helper_function(self, current_node, data):
        if current_node is None:
            return "Cannot predict"
        if current_node.node_index == -1:
            return current_node.node_name
        return self.predict_label_helper_function(current_node.node_children.get(data[current_node.node_index], None), data)


if __name__ == '__main__':
    tree = DecisionTree()
    tree.get_data("dt-data.txt")
    tree.build_decision_tree()
    print("\n<-----------Tree----------->")
    tree.display_tree()

    print("\n\nPrediction for {\"Moderate\", \"Cheap\", \"Loud\", \"City-Center\", \"No\", \"No\"} : ",
          tree.predict_label(["Moderate", "Cheap", "Loud", "City-Center", "No", "No"]))
