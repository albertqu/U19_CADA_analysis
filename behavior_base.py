import numpy as np
import pandas as pd


class EventNode:
    """
    Base Class for behavior log linked list:
    example:
    ------
    from behaviors import BehaviorMat
    code_map = BehaviorMat.code_map
    eventlist = PSENode(None, None, None, None)
    import h5py
    hfile = h5py.File("D1-R35-RV_p155_raw_behavior.mat",'r')
    trial_event_mat = np.array(hfile['out/trial_event_mat'])
    for i in range(len(trial_event_mat)):
        eventcode, etime, trial = trial_event_mat[i, :]
        eventlist.append(PSENode(code_map[eventcode][0] + '|' + code_map[eventcode][1], etime, trial,
        eventcode))
    eventlist.as_df()
    ----
    Now you have a eventlist full of nodes
    call: eventlist.as_df() to get the dataframe
    """
    ABBR = {}

    def __init__(self, event, etime, trial, ecode):
        self.serializable = ['event', 'etime', 'trial', 'ecode']
        if event is None:
            # Implements a circular LinkedList
            self.is_sentinel = True
            self.next = self
            self.prev = self
            self.size = 0
        else:
            self.is_sentinel = False
        self.event = event
        self.etime = etime
        self.trial = trial
        self.ecode = ecode
        # self.trial_start = False # Uncomment this if needed

    def __str__(self):
        if self.is_sentinel:
            return 'Sentinel'
        return f"{type(self).__name__}({self.event}, {self.trial}, {self.etime:.1f}ms, {self.ecode})"

    def trial_index(self):
        # 0.5 is ITI but considered in trial 0
        if self.is_sentinel:
            return None
        else:
            return int(np.ceil(self.trial)) - 1

    # Methods Reserved For Sentinel Node
    def __len__(self):
        assert self.is_sentinel, 'must be sentinel node to do this'
        return self.size

    def __iter__(self):
        assert self.is_sentinel, 'must be sentinel node to do this'
        curr = self.next
        while not curr.is_sentinel:
            yield curr
            curr = curr.next

    def as_df(self, use_abbr=False):
        # Returns an dataframe representation of the information
        assert self.is_sentinel, 'must be sentinel node to do this'
        if use_abbr:
            results = [None] * len(self)
            node_list = self.tolist()
            for i in range(len(self)):
                results[i] = [None] * len(self.serializable)
                for j in range(len(self.serializable)):
                    field = self.serializable[j]
                    attr = getattr(node_list[i], field)
                    results[i][j] = self.ABBR[attr] if attr in self.ABBR else attr
            return pd.DataFrame([[getattr(enode, field) for field in self.serializable] for
                                 enode in self],
                         columns=self.serializable)
        else:
            return pd.DataFrame([[getattr(enode, field) for field in self.serializable] for enode in self],
                                columns=self.serializable)

    def nodelist_asdf(self, nodelist):
        # a method that looks at a restricted view of eventlist
        return pd.DataFrame([[getattr(enode, field) for field in self.serializable] for enode in nodelist],
                                columns=self.serializable)

    # ideally add iter method but not necessary
    def tolist(self):
        assert self.is_sentinel, 'must be sentinel node to do this'
        return [enode for enode in self]

    def append(self, node):
        assert self.is_sentinel, 'must be sentinel node to do this'
        old_end = self.prev
        assert old_end.next is self, "what is happening"
        old_end.next = node
        node.prev = old_end
        self.prev = node
        node.next = self
        self.size += 1
        return node

    def prepend(self, node):
        # Not important
        assert self.is_sentinel, 'must be sentinel node to do this'
        old_first = self.next
        old_first.prev = node
        self.next = node
        node.prev = self
        node.next = old_first
        self.size += 1
        return node

    def remove_node(self, node):
        assert self.is_sentinel, 'must be sentinel node to do this'
        assert self.size, 'list must be non-empty'
        next_node = node.next
        prev_node = node.prev
        prev_node.next = next_node
        next_node.prev = prev_node
        node.next = None
        node.prev = None
        self.size -= 1

    def swap_nodes(self, node1, node2):
        assert self.is_sentinel, 'must be sentinel node to do this'
        assert (not (node1.is_sentinel or node2.is_sentinel)), 'both have to be non-sentinels'
        first_prev = node1.prev
        sec_next = node2.next
        first_prev.next = node2
        node2.prev = first_prev
        node2.next = node1
        node1.prev = node2
        node1.next = sec_next
        sec_next.prev = node1

    def get_last(self):
        assert self.is_sentinel, 'must be sentinel node to do this'
        return self.prev

    def get_first(self):
        assert self.is_sentinel, 'must be sentinel node to do this'
        return self.next

    def is_empty(self):
        assert self.is_sentinel, 'must be sentinel node to do this'
        return self.size == 0


class PSENode(EventNode):
    # Probswitch Event Node
    ABBR = {
        'right': 'RT',
        'left': 'LT',
        'ipsi': 'IP',
        'contra': 'CT',
        'center': 'CE',
    }

    def __init__(self, event, etime, trial, ecode):
        super().__init__(event, etime, trial, ecode)
        self.serializable = self.serializable + ['saliency']
        self.saliency = None
