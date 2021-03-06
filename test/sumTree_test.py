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





class TreeNode():
    def __init__(self, priority, previous):
        self.priority = priority
        self.left = None
        self.right = None
        self.previous = previous  # if previous is None, this node is root




if __name__ == '__main__':

    """
    class Test_Memory(object):
        epsilon = 0.01  # small amount to avoid zero priority
        alpha = 0.6  # [0~1] convert the importance of TD error to priority
        beta = 0.4  # importance-sampling, from initial value increasing to 1
        beta_increment_per_sampling = 0.001
        abs_err_upper = 1.  # clipped abs error

        def __init__(self, capacity):
            self.tree = SumTreeV2(capacity)

            # ========= Memory Operations ============

        def store(self, transition):
            max_p = np.max(self.tree.tree[-self.tree.capacity:])
            if max_p == 0:
                max_p = self.abs_err_upper
            self.tree.add(max_p, transition)  # set the max p for new p

        def enlarge(self, k):
            self.tree.capacity_enlarge(k)

        def sample(self, n):
            b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty(
                (n, self.tree.data[0].size)), np.empty((n, 1))
            pri_seg = self.tree.total_p / n  # priority segment
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

            min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
            for i in range(n):
                a, b = pri_seg * i, pri_seg * (i + 1)
                v = np.random.uniform(a, b)
                idx, p, data = self.tree.get_leaf(v)
                prob = p / self.tree.total_p
                ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
                b_idx[i], b_memory[i, :] = idx, data
            return b_idx, b_memory, ISWeights

        def batch_update(self, tree_idx, abs_errors):
            abs_errors += self.epsilon  # convert to abs and avoid 0
            clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
            ps = np.power(clipped_errors, self.alpha)
            for ti, p in zip(tree_idx, ps):
                self.tree.update(ti, p)

    def gen_t(k):
        transitions = []
        if k == 0:
            transition = np.hstack((0, 0, 0, 0))
            transitions.append(transition)
        for i in range(k):
            s = 1 + i
            a = 2 + i
            r = 3 + i
            s_ = 4 + i
            transition = np.hstack((s, a, r, s_))
            transitions.append(transition)
        return transitions


    test_memory = Test_Memory(50)
    for i in range(50):
        s = 1 + i
        a = 2 + i
        r = 3 + i
        s_ = 4 + i
        ts = np.hstack((s, a, r, s_))
        test_memory.store(ts)
    #test_memory.batch_update()

    test_memory.enlarge(5)

    ts = np.hstack((100, 20, 30, 40))
    for i in range(5):
        ts = np.hstack((100+i, 20+i, 30+i, 40+i))
        test_memory.store(ts)

    b_idx, b_memory, ISWeights = test_memory.sample(10)
    print()
    """


    def test_1():
        root = TreeNode(3, None)
        root.left = TreeNode(1, root)
        root.right = TreeNode(2, root)

        print("Left child's priority: {0}".format(root.left.priority))
        print("Right child's priority: {0}".format(root.right.priority))
        print("Left child backtrack: {0}".format(root.left.previous.priority))
        print("Right child backtrack: {0}".format(root.right.previous.priority))


    def test_2():
        tree = SumTreeV2(5)

        for i in range(5):
            i += 1
            tree.add(p=i, data=i*100)

        tree.capacity_enlarge(6)
        for i in range(6):
            tree.add(p=10 + i, data=20 + i)
        return tree


    tree = test_2()
    #previous_tree = tree
    delete_idx = np.array([7, 8, 10, 11])
    tree.delete(delete_idx)
    print(tree.get_leaves_part())
    tree.capacity_enlarge(7)
    for i in range(7):
        tree.add(p=10 + i, data=20 + i)
    l = tree.get_leaf(4)
    print(l)



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
                delete_idx = np.array([capacity + 1])
                tree.delete(delete_idx)

            print("{0} {1} {2}".format(l, ll, lll))
            count += 1


    #tree = test_2()
    #tree.capacity_enlarge(5)
    #print()


