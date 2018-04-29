import numpy as np
import pymysql as pms


class oraDataFrame(object):
    def __init__(self):
        self.db = self.connectSQL()
        self.cursor = self.db.cursor()

    def connectSQL(self):
        db = pms.Connect("localhost", "root", "Zhang715", "BCW")
        return db

    def createPriorityTable(self, tableName):
        sql = "CREATE TABLE {0}(\
                ID INT PRIMARY KEY NOT NULL AUTO_INCREMENT,\
                STATE_ SMALLINT(5),\
                ACTION_ SMALLINT(5),\
                REWARD_ SMALLINT(5),\
                STATE_NEXT SMALLINT(5),\
                PRIORITY FLOAT )AUTO_INCREMENT=1".format(tableName)
        try:
            self.cursor.execute(sql)
            self.db.commit()
            #print("Already Create TABLE PRIORITY")
        except:
            print("CANNOT CREATE TABLE PRIORITY")
            self.db.rollback()

    def insert(self, transition, priority, tableName):
        """
        Insert transitions and priority
        :param transition: a list of transitions
        :param priority: a list of priorities
        :param tableName:
        :return:
        """
        try:
            for m, p in zip(transition,priority):
                s = int(m[0])
                a = int(m[1])
                r = int(m[2])
                s_ = int(m[3])

                insert = "INSERT INTO %s(\
                             STATE_,\
                             ACTION_,\
                             REWARD_,\
                             STATE_NEXT,\
                             PRIORITY)\
                             VALUES ('%d','%d','%d','%d','%d')" % (tableName, s, a, r, s_,p)
                try:
                    self.cursor.execute(insert)
                    #print("Already insert {0}".format(m))
                except:
                    print("CANNOT INSERT {0}".format(m))
                    self.db.rollback()

            self.db.commit()
        except:
            self.db.rollback()

    def remove(self, tablename, id):
        sql = "DELETE FROM {0} WHERE ID={1}".format(tablename, id)

        try:
            self.cursor.execute(sql)
            self.db.commit()
        except:
            self.db.rollback()
            print("Cannot delete...")

    def cover(self, tablename, id, transition, priority):
        s = int(transition[0])
        a = int(transition[1])
        r = int(transition[2])
        s_ = int(transition[3])
        sql = "UPDATE {0} \
                                  SET STATE_ = {1},  \
                                  ACTION_ = {2},\
                                  REWARD_ = {3},\
                                  STATE_NEXT = {4}, \
                                  PRIORITY = {5} \
                                  WHERE ID = {6}".format(tablename, s, a, r, s_, priority, id)
        try:
            self.cursor.execute(sql)
            self.db.commit()
            # print("Already Update PRIORITY of {0} transition".format(ID))
        except:
            self.db.rollback()
            print("Update Failed!")

    def updatePriority(self, priority, ID, tableName):
        update = "UPDATE {0} SET PRIORITY = {1} WHERE ID = {2}".format(tableName, priority, ID)

        try:
            self.cursor.execute(update)
            self.db.commit()
            #print("Already Update PRIORITY of {0} transition".format(ID))
        except:
            self.db.rollback()
            print("Update Failed!")

    def sumPriority(self, tablename):
        sql = "SELECT SUM(PRIORITY) FROM {0}".format(tablename)

        try:
            self.cursor.execute(sql)
            self.db.commit()
            priority_sum = self.cursor.fetchone()
            return priority_sum[0]
        except:
            self.db.rollback()

    def maxPriority(self, tablename):
        sql = "SELECT MAX(PRIORITY) FROM {0}".format(tablename)

        try:
            self.cursor.execute(sql)
            self.db.commit()
            priority_max = self.cursor.fetchone()
            return priority_max[0]
        except:
            self.db.rollback()

    def get_all_priority(self, tablename):
        sql = "SELECT ID, PRIORITY FROM {0}".format(tablename)

        try:
            self.cursor.execute(sql)
            self.db.commit()
            priorities = self.cursor.fetchall()
            return priorities
        except:
            self.db.rollback()

    def get_row_number(self, tablename):
        sql = "SELECT COUNT(*) FROM {0}".format(tablename)

        try:
            self.cursor.execute(sql)
            self.db.commit()
            rows = self.cursor.fetchone()
            return rows[0]
        except:
            self.db.rollback()

    def extract_transition(self, tablename, id):
        sql = "SELECT STATE_, ACTION_, REWARD_, STATE_NEXT FROM {0} WHERE ID = {1}".format(tablename, id)

        try:
            self.cursor.execute(sql)
            self.db.commit()
            transition = self.cursor.fetchall()
            return transition[0]
        except:
            self.db.rollback()

    def min_idx(self, tablename):
        sql = "SELECT MIN(ID) FROM {0}".format(tablename)
        try:
            self.cursor.execute(sql)
            self.db.commit()
            min_idx = self.cursor.fetchone()
            return min_idx[0]
        except:
            self.db.rollback()


class easySumTree(object):
    def __init__(self):
        self.tableName = 'PRIORITY'
        self.db = oraDataFrame()
        self.create_data_frame()
        self.capacity = 0
        self.tree = None
        self.idframe = None

    def create_data_frame(self):
        self.db.createPriorityTable(self.tableName)

    def add(self, p, transition, id=None):
        """
        :param p:
        :param transition:
        :param id: when the memory is full, the new income transition will be cover the old transition starting from
        first row
        :return:
        """
        if id is None:
            self.db.insert(transition=transition, priority=p, tableName=self.tableName)
        else:
            self.db.cover()

    def remove(self, id):
        self.db.remove(self.tableName, id)

    def update(self, id, priority):
        self.db.updatePriority(priority=priority, ID=id, tableName=self.tableName)

    def max_priority(self):
        return self.db.maxPriority(self.tableName)

    def construct_tree(self):
        self.capacity = self.db.get_row_number(self.tableName)
        self.idframe = np.zeros(self.capacity, dtype=object)
        self.tree = np.zeros(2 * self.capacity - 1)
        priorities = self.db.get_all_priority(self.tableName)
        for i in range(self.capacity):
            id, p = priorities[i]
            self.idframe[i] = id
            tree_idx = i + self.capacity - 1
            self.update_tree(tree_idx, p)

    def update_tree(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

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
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        id_idx = leaf_idx - self.capacity + 1
        id = self.idframe[id_idx]
        transition = self.db.extract_transition(self.tableName, id)
        return id, self.tree[leaf_idx], transition

    def total_p(self):
        if self.tree is not None:
            return self.tree[0]
        else:
            print("Tree is not built...")
            return False

    def clean_tree(self):
        self.tree = None
        self.idframe = None
        self.capacity = None


class pER_Memory(object):

    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.capacity = 0
        self.tree = easySumTree()
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped abs error
        self.data_pointer = 0

    def store(self, transition):
        max_p = self.tree.max_priority()
        if max_p is None:
            max_p = self.abs_err_upper

        if self.capacity < self.max_capacity:
            self.tree.add([max_p], [transition])  # set the max p for new p
            self.data_pointer += 1
            self.capacity += 1
        else:
            # overlap the old transitions
            self.tree.db.cover(self.tree.tableName, self.data_pointer, transition, max_p)
            self.data_pointer += 1

        if self.data_pointer >= self.max_capacity:
            self.data_pointer = self.tree.db.min_idx(self.tree.tableName)

    def sample(self, n):
        # First, construct the sum tree
        self.tree.construct_tree()

        # Initial batch index, batch memory, IS weights
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 4)), np.empty((n, 1))
        pri_seg = self.tree.total_p() / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p()
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data

        # clean the tree
        self.tree.clean_tree()

        return b_idx, b_memory, ISWeights

    def batch_update(self, id, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(id, ps):
            self.tree.update(ti, p)

    def enlarge(self, k):
        self.max_capacity += k
        self.data_pointer = self.capacity  # reset the data pointer back

    def shrink(self, k, removal_id):
        for id in removal_id:
            self.tree.remove(id)
        self.max_capacity -= k
        self.capacity -= k
        self.data_pointer -= k


if __name__ == '__main__':

    def gen_t(k):
        transitions = []
        priorities = []
        if k == 0:
            transition = np.hstack((0, 0, 0, 0))
            transitions.append(transition)
            priorities.append(k)
        for i in range(k):
            s = 1 + i
            a = 2 + i
            r = 3 + i
            s_ = 4 + i
            transition = np.hstack((s, a, r, s_))
            transitions.append(transition)
            priorities.append(i)
        return transitions, priorities

    """ 
    oraDataFrame -- all pass
    db = oraDataFrame()

    db.createPriorityTable('test')

    transition,ps = gen_t(20)
    db.insertMemory(transition, ps,'test')

    print(db.sumPriority('test'))
    priorities = db.get_all_priority('test')
    ID, p = priorities[0]
    print(len(priorities))
    print(ID)
    print(p)
    print(db.get_capacity('test'))
    #db.updatePriority(6,2,'test')
   # print()
    """
    """
    transition, ps = gen_t(20)
    st = easySumTree()
    st.add(ps, transition)
    st.construct_tree()
    print(st.get_leaf(5))
    st.clean_tree()

    for i in range(4):
        st.remove(i)

    st.construct_tree()
    print(st.get_leaf(5))

    idx, batch_memory, transition = st.get_leaf(3)
    print()
    """
    """
    -- Pass
    
    db = oraDataFrame()
    t = db.extract_transition('PRIORITY', 3)
    print()
    """
    """
    transition, ps = gen_t(2)
    db = oraDataFrame()
    #db.remove('PRIORITY', 1)
    db.insert(transition,ps,'PRIORITY')
    """
    """
    db = oraDataFrame()
    #db.createPriorityTable('test')
    #print(db.maxPriority('test'))
    transition, ps = gen_t(0)
    db.insert(transition, ps, 'test')
    """
    """
    db = oraDataFrame()
    #transition, ps = gen_t(0)
    #db.cover('PRIORITY', 4, transition[0], ps[0])
    print(db.min_idx('PRIORITY'))
    print()
    """

    memory = pER_Memory(30)
    transition, ps = gen_t(35)
    for t in transition:
        memory.store(t)  # pass

    tree_idx, batch_memory, ISWeights = memory.sample(5)

    memory.enlarge(5)
    t_, p = gen_t(5)
    for tt in t_:
        memory.store(tt)

    tt_, p_ = gen_t(1)
    for ttt in tt_:
        memory.store(ttt)

    print()

