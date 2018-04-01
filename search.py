class SearchNode(object):
    def __init__(self, data, depth):
        self.data = data
        self.depth = depth
        self.children = []

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

    def get_nodes_by_depth(self, depth):
        nodes = []

        def _get_nodes_by_depth(node):
            if node is not None:
                if node.depth == depth:
                    nodes.append(node)
                for n in node.children:
                    _get_nodes_by_depth(n)

        _get_nodes_by_depth(self)
        return nodes
