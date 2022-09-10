# author: Lexi Zhou

import classes as cl
import copy


class DoublyLinkedList:
    """A doubly linked list data structure"""

    def __init__(self):
        self.sentinel = cl.Event_Node()
        self.sentinel.next = None
        self.sentinel.prev = None
        self.size = 0

    def add_first(self, item):
        new_item = copy.deepcopy(item)
        self.sentinel.next = new_item
        self.sentinel.prev = new_item
        new_item.next = self.sentinel
        new_item.prev = self.sentinel
        self.size += 1

    def add_to_end(self, item):
        if self.size == 0:
            self.add_first(item)
            return
        new_last = copy.deepcopy(item)
        self.sentinel.prev.next = new_last
        new_last.prev = self.sentinel.prev
        self.sentinel.prev = new_last
        new_last.next = self.sentinel
        self.size += 1

    def add_to_start(self, item):
        if self.size == 0:
            self.add_first(item)
            return
        new_first = copy.deepcopy(item)
        self.sentinel.next.prev = new_first
        new_first.next = self.sentinel.next
        self.sentinel.next = new_first
        new_first.prev = self.sentinel
        self.size += 1

    def swap_nodes(self, a, b):
        assert (a.next is b) and (b.prev is a)
        up = a.prev
        down = b.next
        up.next = b
        down.prev = a
        a.next = down
        b.prev = up
        a.prev = b
        b.next = a
