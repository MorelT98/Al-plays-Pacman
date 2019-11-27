class Stack:
    def __init__(self, CAPACITY=1024):
        self.array = [0] * CAPACITY
        self.top = -1
        self._initial_capacity = CAPACITY
        self._current_capacity = CAPACITY

    def push(self, val):
        if self.is_full():
            self._increase_capacity()
        self.top += 1
        self.array[self.top] = val

    def pop(self):
        if self.top < 0:
            raise Exception('The stack is empty')
        val = self.array[self.top]
        self.array[self.top] = 0
        self.top -= 1
        return val

    def peek(self):
        if self.top < 0:
            raise Exception('The stack is empty')
        return self.array[self.top]

    def is_empty(self):
        return self.top == -1

    def is_full(self):
        return self.top + 1 == self._current_capacity

    def size(self):
        return self.top + 1

    def __str__(self):
        text = '['
        for i in range(0, self.top + 1):
            text += '{}|'.format(self.array[i])
        return text

    def _increase_capacity(self):
        self.array += [0] * self._initial_capacity
        self._current_capacity += self._initial_capacity


class Queue:
    def __init__(self, CAPACITY=1024):
        self.array = [0] * CAPACITY
        self.head = 0
        self.tail = -1
        self._initial_capacity = CAPACITY
        self._current_capacity = CAPACITY

    def enqueue(self, val):
        if self.is_full():
            print('Reached full capacity!')
            self._increase_capacity()
        self.tail += 1
        self.array[self.tail] = val

    def dequeue(self):
        val = self.array[self.head]
        self.array[self.head] = 0
        self.head += 1
        return val

    def is_empty(self):
        return self.tail == self.head - 1

    def is_full(self):
        return self.tail + 1 >= self._current_capacity

    def size(self):
        return self.tail - self.head + 1

    def __str__(self):
        text = ''
        text += '   '
        text += '-' * self.size() * 16
        text += '\n'
        text += '<-- '
        for i in range(self.head, self.tail):
            text += '\t{}\t|'.format(self.array[i])
        text += '\t{}\t'.format(self.array[self.tail])
        text += ' <--'
        text += '\n'
        text += '   '
        text += '-' * self.size() * 16
        return text

    def _increase_capacity(self):
        # first, move the elements from the current head to zero, to use allocated space
        if self.head != 0:
            for i in range(self.head, self.tail + 1):
                self.array[i - self.head] = self.array[i]
                self.array[i] = 1
            self.tail -= self.head
            self.head = 0

        # if the array is still full, increase allocated space
        if self.is_full:
            self.array += [0] * self._initial_capacity
            self._current_capacity += self._initial_capacity

class Heap:
    '''
    This is an implementation of the Priority Queue ADT called Heap. A heap is a specialized tree
    satisfying the following property: In a max heap, for any given node C, if P is a parent of
    node C, then the value of P is grater or equal to the value of C; In a min heap, the value of
    P is smaller or equal to the value of C.

    This is an implementation of a binary min heap.
    '''

    class Node:
        def __init__(self, val, idx):
            self.val = val
            self.idx = idx

        # Comparisons necessary to add the node into a heap
        def __gt__(self, other):
            return self.val > other.val
        def __ge__(self, other):
            return self.val >= other.val
        def __lt__(self, other):
            return self.val < other.val
        def __le__(self, other):
            return self.val <= other.val
        def __eq__(self, other):
            return self.val == other.val
        def __str__(self):
            return str(self.val)

    def __init__(self, initial_capacity=32):
        self.__capacity__ = initial_capacity    # capacity of the underlying array
        self.__expand_factor__ = 1.8   # How much we should expand the array if full
        self.__array__ = [self.Node(0, i) for i in range(initial_capacity)]  # underlying array
        self.size = 0
        self.root = None

    def get(self, idx):
        '''Returns the node at index idx'''
        if idx >= self.size:
            return
        return self.__array__[idx]

    def is_empty(self):
        '''Returns True if the heap is empty, False if not'''
        return self.size == 0

    def left(self, index):
        '''Returns the index of the left child of the node at the given index'''
        return 2 * index + 1

    def right(self, index):
        '''Returns the index of the right child of the node at the given index'''
        return 2 * index + 2

    def parent(self, index):
        '''Returns the index of the parent of the node at the given index'''
        return (index - 1) // 2

    def add(self, val):
        '''Inserts the given value into the heap'''
        # Check for capacity
        if self.size >= self.__capacity__:
            self.__increase_capacity__()

        # Add element at the end
        self.__array__[self.size] = self.Node(val, self.size)

        # Maintain the heap property
        i = self.size
        parent = self.parent(i)
        while i > 0 and self.__array__[i] < self.__array__[parent]:
            # Repeatedly swap node with its parent, until its value is no longer smaller than it's parent's
            self.__array__[i], self.__array__[parent] = self.__array__[parent], self.__array__[i]
            # update indices
            self.__array__[i].idx = parent
            self.__array__[parent].idx = i
            # move up the heap
            i = parent
            parent = self.parent(i)

        if self.size == 0:
            self.root = self.__array__[0]
        self.size += 1

    def __increase_capacity__(self):
        '''Increases the capacity of the underlying array when it gets full'''
        new_capacity = int((self.__capacity__ * self.__expand_factor__))
        new_array = [self.Node(0, 0) for i in range(new_capacity)]
        for i in range(self.size):
            new_array[i] = self.__array__[i]
        # Update array
        self.__array__ = new_array
        # Update capacity
        self.__capacity__ = new_capacity

    def add_all(self, collection):
        '''Add all the values in the given collection into the heap'''
        for val in collection:
            self.add(val)

    def remove(self):
        '''Removes and returns the root of the heap, or None if the heap is empty'''
        if self.size == 0:
            return
        node = self.__array__[0]

        # Replace the root's value with the last element in the heap
        self.__array__[0] = self.__array__[self.size - 1]

        # Maintain the heap property
        i = 0
        while i < self.size:
            # Repeatedly swap node with the smallest of its children, until the node is no longer smaller
            # than both of its children
            if self.left(i) < self.size:
                min_idx = self.left(i)
                # Only check for the right child if it's in the bounds
                if self.right(i) < self.size:
                    if self.__array__[self.right(i)] < self.__array__[self.left(i)]:
                        min_idx = self.right(i)
                if self.__array__[i] < self.__array__[min_idx]:
                    break
                else:
                    self.__array__[i], self.__array__[min_idx] = self.__array__[min_idx], self.__array__[i]
                    i = min_idx
            else:
                break
        self.size -= 1
        if self.size == 0:
            self.root = None

        return node.val


    def __str__(self):
        return str(list(map(lambda x:x.val, self.__array__[0:self.size])))