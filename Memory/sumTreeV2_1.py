import numpy as np


class SumTreeV2(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """

    # TODO: Under construction
    # TODO: STILL cannot figure out the fatal problem of deletion
    # TODO: How to keep the structure of the tree after delection

    data_pointer = 0

    def __init__(self, capacity):
        self.last_capacity = capacity
        self.const_last_capacity = self.last_capacity
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity

    def add(self, p, data):
        pd = np.hstack((p, data))
        if self.data_pointer >= self.last_capacity and self.last_capacity < self.capacity:
            tree_idx = self.data_pointer + self.last_capacity
            self.data[self.data_pointer] = data  # update data_frame
            self.update(tree_idx, p)  # update tree_frame

            self.data_pointer += 1
            self.last_capacity += 1

            if self.data_pointer == self.capacity:
                if self.capacity > self.const_last_capacity:
                    delta = self.capacity - self.const_last_capacity
                    self.synchronize(delta)

                elif self.capacity < self.const_last_capacity:
                    delta = self.const_last_capacity - self.capacity
                    self.synchronize(delta)

                # when we enlarge the tree, the original order will be disorganize
                # the index of data does not match the index of the tree
                # thus, we need to update the original data frame with the new order stored in the new tree
                # only cope the leaf; i.e. from N-1 to the last one

        else:
            """
            if self.data_pointer >= self.capacity:  # replace when exceed the capacity
                for i in range(len(self.data)):
                    if self.data[i] != 0:
                        continue
                    else:
                        self.data[i] = data
                        tree_idx = (self.capacity - 1) + i
                        self.update(tree_idx, p)
                        break
                print("--> the memory is full")
            """

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
            self.tree = np.insert(self.tree, insert_pos, [left, right])

            # Data frame's order must be change too
            idx += 1
            count += 1

        self.last_capacity = self.capacity  # mark down the last capacity for adding operation
        self.const_last_capacity = self.last_capacity  # unchange
        self.capacity += k  # Update the value of capacity
        self.data = np.insert(self.data, self.data.shape[0], np.zeros(k))  # The data frame also need to be extended

        # synchronize the data frame after enlarge because of the change of the tree structure
        #self.synchronize(k)

    def synchronize(self,k):
        """
        synchronize the data frame after enlarge because of the change of the tree structure
        :param k: the increment k
        :return:
        """
        # synchronize the data frame after enlarge because of the change of the tree structure
        tmp_data = np.delete(self.data, range(k))  # first delete first k elements in the data frame
        original_tree_idx = self.const_last_capacity - 1  # get the old tree index of the first element in the data frame
        for i in range(k):
            changed_tree_idx = original_tree_idx * 2 + 1  # the current tree index of the first element of data frame
            change_to_data_idx = changed_tree_idx - self.capacity + 1  # should be change the element to this data index
            original_data_idx = i

            tmp_data = np.insert(tmp_data, change_to_data_idx, self.data[original_data_idx])
            original_tree_idx += 1

        self.data = tmp_data

    def delete(self, leaf_nodes):
        num_delete = leaf_nodes.shape[0]
        lp = self.get_leaves_part()
        del_idx = leaf_nodes - (self.capacity - 1)

        lp = np.delete(lp, del_idx)

        self.last_capacity = self.capacity
        self.const_last_capacity = self.last_capacity
        self.capacity -= num_delete
        self.data_pointer -= num_delete

        self.reconstruct(self.capacity, lp)

    def reconstruct(self, capacity, leave_part):
        """
        reconstruct the tree
        :return:
        """
        self.tree = np.zeros(2 * capacity - 1)
        for l in range(leave_part.shape[0]):
            leaf = leave_part[l]
            tp = l + capacity - 1
            self.update(tp, leaf)

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
        self.const_last_capacity = self.last_capacity
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

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def get_leaves_part(self):
        lp = self.tree[self.capacity-1:]
        return lp

    @property
    def total_p(self):
        return self.tree[0]  # the root

