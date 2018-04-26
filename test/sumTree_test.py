import numpy as np


class SumTreeV2(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.last_capacity = capacity
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        if self.data_pointer >= self.last_capacity and self.last_capacity < self.capacity:
            tree_idx = self.data_pointer + self.last_capacity
            self.data[self.data_pointer] = data  # update data_frame
            self.update(tree_idx, p)  # update tree_frame

            self.data_pointer += 1
            self.last_capacity += 1

        else:
            if self.data_pointer >= self.capacity:  # replace when exceed the capacity
                print("--> the memory is full")
                return

            tree_idx = self.data_pointer + self.capacity - 1
            self.data[self.data_pointer] = data  # update data_frame
            self.update(tree_idx, p)  # update tree_frame

            self.data_pointer += 1

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2  # // 向下取整
            self.tree[tree_idx] += change

    def capacity_enlarge(self, k):
        """
        Increase to N + k
        :param k:
        :return:
        """
        count = 0
        idx = self.capacity - 1
        while count < k:
            left = self.tree[idx]
            right = 0
            insert_pos = self.tree.shape[0]
            self.tree = np.insert(self.tree, insert_pos, [left,right])
            idx += 1
            count += 1

        self.last_capacity = self.capacity  # mark down the last capacity for adding operation
        self.capacity += k  # Update the value of capacity
        self.data = np.insert(self.data, self.data.shape[0], np.zeros(k))  # The data frame also need to be extended

    def delete_node(self, del_nodes):
        """
        Delete k specific node
        :param: del_nodes: a list of nodes which is to be deleted
        :return:
        """
        del_total = len(del_nodes)



    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class TreeNode():
    def __init__(self, priority, previous):
        self.priority = priority
        self.left = None
        self.right = None
        self.previous = previous  # if previous is None, this node is root




if __name__ == '__main__':
    def test_1():
        root = TreeNode(3, None)
        root.left = TreeNode(1, root)
        root.right = TreeNode(2, root)

        print("Left child's priority: {0}".format(root.left.priority))
        print("Right child's priority: {0}".format(root.right.priority))
        print("Left child backtrack: {0}".format(root.left.previous.priority))
        print("Right child backtrack: {0}".format(root.right.previous.priority))

    def test_2():
        tree = SumTreeV2(4)

        for i in range(4):
            i += 1
            tree.add(p=i,data=i)

        tree.capacity_enlarge(10)
        for i in range(10):
            tree.add(p=10+i, data=10+i)
        return tree


    tree = test_2()