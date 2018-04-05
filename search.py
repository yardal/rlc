from fen import *

class SearchNode(object):
    def __init__(self, data, depth):
        self.data = data
        self.depth = depth
        self.children = []

    def __str__(self):

        string = " " * self.depth + " Data: " + self.data + " number of Children " + str(len(self.children)) + "\n"
        for child in self.children:
            string = string + str(child)

        return string

    def add_child(self, obj):
        self.children.append(obj)

    def size(self):
        if self.children == []:
            return 1

        size_ = 0
        for j in self.children:
            size_ = size_ + j.size()
        return size_

    def get_leaf_nodes(self):
        leafs = []

        def _get_leaf_nodes(node):
            if node is not None:
                if len(node.children) == 0:
                    leafs.append(node)
                for n in node.children:
                    _get_leaf_nodes(n)

        _get_leaf_nodes(self)
        return leafs


def alpha_beta_search(node, depth, alpha, beta, maximizer, net):
    if depth == 0 or node.children == []:
        return net.eval(get_tensor_from_FEN(node.data))

    if maximizer:
        v = -4.0
        for child in node.children:
            v = max(v, alpha_beta_search(child, depth - 1, alpha, beta, not maximizer, net))
            alpha = max(alpha, v)
            if beta <= alpha:
                break
            return v
    else:
        v = 4.0
        for child in node.children:
            v = min(v, alpha_beta_search(child, depth - 1, alpha, beta, maximizer, net))
            alpha = min(beta, v)
            if beta <= alpha:
                break
            return v
