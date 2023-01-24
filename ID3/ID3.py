import math

import numpy as np

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names  # we didnt use it - need to think about it
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()  # TO ASK
        self.min_for_pruning = min_for_pruning  # TO ASK

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        for count in counts.values():
            p = count / rows.shape[0]
            impurity -= (p * math.log(p, 2))
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        left_uncertainty = ID3.entropy(left, left_labels)
        right_uncertainty = ID3.entropy(right, right_labels)
        info_gain_value = current_uncertainty - ((len(left) / (len(left) + len(right))) * left_uncertainty) - (
                    (len(right) / (len(left) + len(right))) * right_uncertainty)
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        true_rows = []
        true_labels = []
        false_rows = []
        false_labels = []
        for row, label in zip(rows, labels):
            if question.match(row):
                true_rows.append(row)
                true_labels.append(label)
            else:
                false_rows.append(row)
                false_labels.append(label)
        gain = self.info_gain(np.array(false_rows), np.array(false_labels), np.array(true_rows), np.array(true_labels),
                              current_uncertainty)
        # ========================

        return gain, np.array(true_rows), np.array(true_labels), np.array(false_rows), np.array(false_labels)

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======
        for f_idx in range(rows.shape[1]):
            sorted_by_f = [row[f_idx] for row in rows]
            sorted_by_f.sort()
            for i in range(len(sorted_by_f) - 1):
                t_i = (sorted_by_f[i] + sorted_by_f[i + 1]) / 2
                question = Question([row[f_idx] for row in rows], f_idx, t_i)
                gain, true_rows, true_labels, false_rows, false_labels = self.partition(rows, labels, question,
                                                                                        current_uncertainty)
                if gain >= best_gain:
                    best_gain = gain
                    best_true_rows = true_rows
                    best_true_labels = true_labels
                    best_question = question
                    best_false_rows = false_rows
                    best_false_labels = false_labels

        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        my_labels = set(labels)
        if len(set(labels)) == 1:
            return Leaf(rows, labels)
        elif len(rows) <= self.min_for_pruning:
            counts = class_counts(rows, labels)
            if (counts["M"] >= counts["B"]):
                for label in labels:
                    if label == "B":
                        label = "M"
                return Leaf(rows, labels)
            else:
                for label in labels:
                    if label == "M":
                        label = "B"
                return Leaf(rows, labels)
        else:
            best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = self.find_best_split(
                np.array(rows), np.array(labels))
            true_branch = self.build_tree(best_true_rows, best_true_labels)
            false_branch = self.build_tree(best_false_rows, best_false_labels)

        # ========================
        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        # TODO: check if x_train y_train is just rows, labels?
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        if isinstance(node, Leaf):
            prediction = list(node.predictions.keys())[0]  # TODO: maybe needs to change??
        else:  # node is DecisionNode
            if (node.question.match(row)):
                prediction = self.predict_sample(row, node.true_branch)
            else:
                prediction = self.predict_sample(row, node.false_branch)
        # ========================

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = None

        # ====== YOUR CODE: ======
        y_pred = []
        for row in rows:
            y_pred.append(self.predict_sample(row))
        # ========================

        return np.array(y_pred)


'''    class Prediction:
        def __init__(self, x_test):
            self.x_test = x_test
            self.y_pred = None

        def predict(self):
            self.y_pred = ID3.predict(self.x_test)'''
