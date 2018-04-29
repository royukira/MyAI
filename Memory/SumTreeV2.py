"""

The sumTree version 2

This version is suitable for the apER algorithm which need a flexible data structure to support frequent update operation

"""

import numpy as np


class priorityNode(object):
    def __init__(self, priority, transition):
        self.priority = priority
        self.transition = transition


class SumTreeV2(object):
    """
    The SumTreeV2 is developed based on the SumTree version 1
    The code of SumTree version 1 is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Store the data with it priority in tree and data frameworks.
    """

    # TODO: Under construction
    # TODO: STILL cannot figure out the fatal problem of deletion
    # TODO: How to keep the structure of the tree after delection

    data_pointer = 0

    def __init__(self, capacity):
        self.last_capacity = capacity
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        #                    int                                     node
        #

    def add(self, p, data):
        pNode = priorityNode(p, data)
        if self.data_pointer >= self.last_capacity and self.last_capacity < self.capacity:
            tree_idx = self.data_pointer + self.last_capacity
            self.update(tree_idx, pNode)  # update tree_frame

            self.data_pointer += 1
            self.last_capacity += 1

        else:
            """
            if self.data_pointer >= self.capacity:  # replace when exceed the capacity
                for i in range(len(self.data)):
                    if self.data[i] != 0:
                        continue
                    else:
                        self.data[i] = data
                        tree_idx = (self.capacity-1) + i
                        self.update(tree_idx, p)
                        break
                print("--> the memory is full")
            """
            tree_idx = self.data_pointer + self.capacity - 1
            self.update(tree_idx, pNode)  # update tree_frame
            self.data_pointer += 1

    def update(self, tree_idx, p):
        if self.tree[tree_idx] != 0:
            change = p.priority - self.tree[tree_idx]
        else:
            change = p.priority - self.tree[tree_idx].priority
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2  # // 向下取整
            self.tree[tree_idx] += change   # [ N - 1 ] 部分都是 int

    def reconstruct(self, capacity):
        """
        reconstruct the tree
        :return:
        """
        def check(data):
            if capacity != len(data):
                raise Exception(" The data frame is broken...")

        # CHECK IF THE DATA FRAME GO WRONG
        try:
            check(self.data)
        except Exception as err:
            print(1, err)

        self.tree = np.zeros(2 * capacity - 1)
        for dp in range(self.data.shape[0]):
            data = self.data[dp]
            tp = dp + capacity - 1
            self.update(tp, data)

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
            right = priorityNode(0, None)
            insert_pos = self.tree.shape[0]
            self.tree = np.insert(self.tree, insert_pos, [left,right])
            idx += 1
            count += 1

        self.last_capacity = self.capacity  # mark down the last capacity for adding operation
        self.capacity += k  # Update the value of capacity

    def delete(self, leaf_nodes):
        num_delete = leaf_nodes.shape[0]

        self.last_capacity = self.capacity
        self.capacity -= num_delete
        self.data_pointer -= num_delete

        self.reconstruct(self.capacity)

    def delete_node(self, leaf_nodes):
        # TODO: Under construction
        # TODO: how to keep the structure of the tree after deletion
        # TODO: How about reconstruct the tree? using the data frame?
        """
        Delete k specific node
        :param: del_nodes: an array of nodes which is to be deleted
        :return:
        """
        num_delete = leaf_nodes.shape[0]
        del_data_idx = leaf_nodes - (self.capacity - 1)
        all_delete_idx = np.copy(leaf_nodes)  # save all nodes which need to be deleted

        # Keep the tree structure as [ N -1 | N ]
        # no matter how to delete
        # Now, we have to create a list of deleting nodes
        for idx in leaf_nodes:
            if idx % 2 != 0:
                parent_idx = (idx - 1) // 2
                if (idx + 1) in leaf_nodes:

                    # idx is left child and (idx+1) is right child;
                    # if both are deleted, partent have to be delete as well

                    all_delete_idx = np.append(all_delete_idx, parent_idx)

                    # When parent is going to be deleted, their brother need to be deleted as well
                    if parent_idx % 2 != 0:
                        right_parent_idx = parent_idx + 1
                        all_delete_idx = np.append(all_delete_idx, right_parent_idx)
                    elif parent_idx % 2 == 0:
                        left_parent_idx = parent_idx - 1
                        all_delete_idx = np.append(all_delete_idx, left_parent_idx)
                else:
                    right_idx = idx + 1

                    # If the left child is deleted, the right child should be deleted as well;
                    # Then the value of the right child will be assign to its parent
                    # The value of their parent is equal to the value of right child after the left being delete
                    # Thus, the right node is useless and should be deleted

                    all_delete_idx = np.append(all_delete_idx, right_idx)

                    # Update the value of parent; because the left node is deleted
                    # tree[parent_idx] =  tree[parent_idx] - tree[idx]
                    while parent_idx != 0:
                        # propagate to the root
                        self.tree[parent_idx] -= self.tree[idx]
                        parent_idx = (idx - 1) // 2

            elif idx % 2 == 0:
                parent_idx = (idx - 1) // 2
                left_idx = idx - 1
                all_delete_idx = np.append(all_delete_idx, left_idx)
                while parent_idx != 0:
                    # propagate to the root
                    self.tree[parent_idx] -= self.tree[idx]
                    parent_idx = (idx - 1) // 2

        # Start to delete
        self.tree = np.delete(self.tree, all_delete_idx)
        self.data = np.delete(self.data, del_data_idx)

        # Update parameters
        self.last_capacity = self.capacity
        self.capacity -= num_delete
        self.data_pointer -= num_delete

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
        def check():
            if self.data_pointer != self.data.shape[0]:
                raise Exception("Learning and memory adjustment MUST be synchronous. \
                \ni.e.learn and adjust in every k step ")

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
        try:
            # check whether training and memory adjustment
            check()
        except Exception as err:
            print(1, err)


        return leaf_idx, self.tree[leaf_idx], self.tree[leaf_idx].transition

    @property
    def total_p(self):
        return self.tree[0]  # the root

if __name__ == '__main__':

    def test2():
        tree = SumTreeV2(5)

        for i in range(6):
            i += 1
            tree.add(p=i, data=i)

        tree.capacity_enlarge(10)
        for i in range(12):
            tree.add(p=10 + i, data=10 + i)
        return tree


    def test3():
        tree = SumTreeV2(5)
        for i in range(5):
            i += 1
            tree.add(p=i, data=i)

        count = 0

        while count < 20:
            l, ll, lll = tree.get_leaf(4)
            choice = np.random.uniform()
            if count % 4 == 0:
                tree.capacity_enlarge(5)
                for i in range(5):
                    tree.add(p=10 + i, data=10 + i)
            else:
                capacity = tree.capacity
                delete_idx = np.array([capacity+1])
                tree.delete(delete_idx)

            print("{0} {1} {2}".format(l,ll,lll))
            count += 1


    tree = test2()
