
import numpy as np
import random

# ========= Node ===========


class TimeNode(object):
    """
    For store the transitions generated at ith time step (i.e. self.time = i )
    """
    def __init__(self, time):
        self.previous = None
        self.time = time
        self.transitions = []
        self.nextTime = None

    def get_time(self):
        return self.time

    def next_time(self):
        return self.nextTime

    def extract_transition(self, idx):
        """
        Extract transitions from the list
        :param idxs: the indies of transitions
        :return:
        """
        return self.transitions[idx]

    def random_extract(self, k):
        """
        Randomly extract k transitions that is stored in the stack of transitions
        :param k:
        :return: a list of transitions
        """
        transitions = []
        for i in range(k):
            t = self.transitions.pop(random.randrange(len(self.transitions)))  # the sampled transitions must be pop out
            transitions.append(t)
        #t = random.sample(self.transitions, k)
        return transitions

    def delete_transition(self, idx):
        """
        Pop the transitions from the list
        :param idxs: the indies of transitions
        :return:
        """
        pop_out = self.transitions.pop(idx)

        return pop_out

    def insert_transition(self, transition):
        """
        Insert the transition
        :param transition:
        :return:
        """
        Tid = [self.time, len(self.transitions)]  # the time tag
        transition = np.append(transition, Tid)
        self.transitions.append(transition)
        return transition

    def transition_length(self):
        """
        get the length of transition
        :return:
        """
        return len(self.transitions)

    def is_empty(self):
        """
        Check whether the transition set is empty
        :return:
        """
        if len(self.transitions) == 0:
            return True
        else:
            return False


# ========= hash ===========


class subMap(object):
    def __init__(self):
        self.items = []

    def add(self, k, v):
        self.items.append((k, v))

    def get_node(self, k):
        for key, val in self.items:
            if key == k:
                return val
        return False  # 找不到就return False

    def remove(self, k):
        for key, val in self.items:
            if key == k:
                self.items.remove((key, val))

    def is_transition_empty(self):
        if len(self.items) == 0:
            return True  # item里的 time node 已被删除

        time_node = self.items[0][1]
        transitionSet = time_node.transitions
        for i in range(len(transitionSet)):
            if transitionSet[i] is not None:
                return False  # 没有全部为None
            else:
                continue
        return True


class hashMap(object):
    def __init__(self, n):
        self.hmaps = []
        for i in range(n):
            self.hmaps.append(subMap())

    def map_idx(self, time):
        """
        Using hash to find the correspondent sub-map
        :param time:
        :return:
        """
        idx = hash(time) % len(self.hmaps)
        return self.hmaps[idx]

    def add(self, time, node):
        """
        Add a node to a correspondent sub-map
        :param time:
        :param node:
        :return:
        """
        sub_map = self.map_idx(time)
        sub_map.add(time, node)

    def remove(self, time):
        """
        Remove the node stored in the sub-map sub_map
        :param time:
        :return:
        """
        sub_map = self.map_idx(time)
        sub_map.remove(time)
        """
        // cannot delete the sub-map, otherwise it will cause chaos; the time cannot match the correspondent sub-map 
        if sub_map.is_empty():
            idx = hash(time) % len(self.hmaps)
            self.remove_submap(idx)
        """

    def remove_submap(self, hmap_idx):
        """
        Remove the sub-map
        :param hmap_idx:
        :return:
        """
        self.hmaps.pop(hmap_idx)

    def get(self, time):
        """
        Get the node of the specific time
        :param time:
        :return:
        """
        idx = self.map_idx(time)
        return idx.get_node(time)

    def check_empty(self,time):
        """
        Check if the specific sub-map is empty
        :param time:
        :return:
        """
        sub_map = self.map_idx(time)
        if sub_map.is_transition_empty():
            return True
        else:
            return False


class timeTable(object):
    def __init__(self):
        self.maps = hashMap(2)
        self.num = 0

    def get_node(self, time):
        """
        get the time node
        :param time:
        :return:
        """
        return self.maps.get(time)

    def add(self, time, node):
        if self.num == len(self.maps.hmaps):
            self.resize()

        self.maps.add(time, node)
        self.num += 1

    def remove(self, time):
        self.maps.remove(time)
        self.num -= 1

    def resize(self):
        new_maps = hashMap(self.num * 2)

        for m in self.maps.hmaps:
            for k, v in m.items:
                new_maps.add(k, v)

        self.maps = new_maps

    def find_oldest(self):
        """
        Find the first non-empty hmaps and return the node stored in the hmaps
        :return: the sub-map item -- [(time, time_node)]
        """
        for i in range(len(self.maps.hmaps)):
            if self.maps.hmaps[i].is_transition_empty():
                continue
            else:
                return self.maps.hmaps[i]

    def k_oldest(self, k):
        """
        Find k oldest transitions
        :param k:
        :return: a list of oldest transitions
        """
        oldest_time_node = None

        oldest_item = self.find_oldest()
        for kk, v in oldest_item.items:
            _ = kk  # the time id; useless for now
            oldest_time_node = v

        transition_length = oldest_time_node.transition_length()

        if transition_length >= k:
            # we can sample k oldest transitions in one time node
            oldest_transitions = oldest_time_node.random_extract(k)
        else:
            rest = k - transition_length
            oldest_transitions = oldest_time_node.random_extract(transition_length)
            # Find the rest transitions in the next time node
            # 一直搜索到够k个为止
            rest_transitions = []
            while True:
                next_time_node = oldest_time_node.nextTime  # next time node
                next_t_length = next_time_node.transition_length()  # the length of transition set in the next time node
                if next_t_length >= rest:
                    rest_transitions = next_time_node.random_extract(rest)
                    for r in rest_transitions:
                        oldest_transitions.append(r)
                    break
                else:
                    rest_transitions.append(next_time_node.random_extract(next_t_length))
                    rest -= next_t_length

        return oldest_transitions


# ====== Time tag memory =====

class timeTag(object):
    def __init__(self):
        self.head = TimeNode(None)
        self.length = 0
        self.current = self.head
        self.timeTable = timeTable()

    def add_node(self, time, transitions):
        """
        Add the time node to the time chain and the time hash table (the correspondent sub-map)
        :param time: ith time step
        :param transitions: a list of transitions generated at ith time step
        :return:
        """
        # Preprocess -- assign the Time ID
        for t in range(len(transitions)):
            tid = [time, t]
            transitions[t] = np.append(transitions[t], tid)

        new_node = TimeNode(time)
        new_node.transitions = transitions
        if self.is_empty():
            self.head.nextTime = new_node
            new_node.previous = self.head
            self.current = new_node
            self.timeTable.add(time, new_node)  # add to the hash table
        else:
            self.current.nextTime = new_node
            new_node.previous = self.current
            self.current = new_node
            self.timeTable.add(time, new_node)  # add to the hash table

        self.length += 1

    def get_node(self, time):
        return self.timeTable.get_node(time)

    def insert_transition(self, transition):
        """
        Insert the transition into the latest time node (i.e. self.current)
        Note: self.current points at the latest time node
        :param transition:
        :return:
        """
        transition = self.current.insert_transition(transition)
        return transition

    def extract_transition(self, T, idx):
        """
        Extract the specific transition at T time step

        If T is current time, directly return the transition using index 'idx'

        Else search the time hash table to find T's node, and get the transition in that node using index 'idx'

        :param T: the specific time step
        :param idx: the index of the transition in the T time node
        :return: A transition
        """
        if T == self.current.time:
            transition = self.current.extract_transition(idx)
        else:
            T_node = self.timeTable.get_node(T)

            if T_node is False:
                print('Cannot find the time node of T')
                return

            transition = T_node.extract_transition(idx)
        return transition

    def remove_transition(self, T, idx):
        """
        Remove the transition from the specific T time node
        :param T: the specific time step
        :param idx: the index of transition stored at the T time node
        :return: None
        """
        if T == self.current.time:
            self.current.delete_transition(idx)
        else:
            T_node = self.timeTable.get_node(T)
            if T_node is False:
                print('Cannot find the time node of T')
            #T_node.delete_transition(idx)
            T_node.transitions[idx] = None

    def remove_node(self, T):
        """
        Remove the T time node from the hash table

        Note: only remove the node stored in the sub-map; however, the sub-map is reserved

        :param T: the specific time node
        :return: None
        """
        T_node = self.timeTable.get_node(T)

        previous_node = T_node.previous
        next_node = T_node.nextTime

        if next_node == None:
            # That means the previous_node become the last node in the chain
            previous_node.nextTime = None
        else:
            previous_node.nextTime = next_node
            next_node.previous = previous_node

        self.timeTable.remove(T)
        self.length -= 1

    def select_k_oldest(self, k):
        """
        Select K oldest transitions
        :param k:
        :return: a list of oldest transitions
        """
        oldest_transitions = self.timeTable.k_oldest(k)
        return oldest_transitions

    def is_empty(self):
        """
        Check whether the time chain is empty
        :return:
        """
        if self.length == 0:
            return True
        else:
            return False



if __name__ == '__main__':
    """
    tc = timeChainV1()
    tc.add(1)
    tc.add(2)
    print()
    
    tb = timeTable()
    for i in range(100):
        tb.add(i, i)
    tb.remove(3)
    print()"""

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

    tag = timeTag()
    t = 30
    for tt in range(t):
        ts = gen_t(tt)
        tag.add_node(tt, ts)  # store the transition

    new_ts = gen_t(1)
    tag.insert_transition(new_ts)   # pass  // for storing transition
    tag.insert_transition(new_ts)   # pass
    ts_ = tag.extract_transition(28, 23)  # pass  // for sample the transition
    tag.remove_transition(28,23)  # pass // for remove the transition
    tag.remove_transition(0,0)  # pass
    tag.remove_transition(1,0)  # pass
    check = tag.get_node(1)  # pass // get a time node in T time step
    tag.remove_node(2)
    oldest = tag.timeTable.find_oldest()  # pass  // find the oldest sub-map item -- [(time, time_node)]
    oldest_ts = tag.select_k_oldest(4)  # pass  // find the k oldest transitions
    tag.remove_node(29)  # pass  // the sub-map is empty; but the sub-map item is reversed
    test_get_no = tag.get_node(40)
    print(test_get_no)