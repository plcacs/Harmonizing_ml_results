
import pyperf
import gc
CYCLES = 100
LINKS = 20

class Node():

    def __init__(self):
        self.next = None
        self.prev = None

    def link_next(self, next):
        self.next = next
        self.next.prev = self

def create_cycle(node, n_links):
    'Create a cycle of n_links nodes, starting with node.'
    if (n_links == 0):
        return
    current = node
    for i in range(n_links):
        next_node = Node()
        current.link_next(next_node)
        current = next_node
    current.link_next(node)

def create_gc_cycles(n_cycles, n_links):
    'Create n_cycles cycles n_links+1 nodes each.'
    cycles = []
    for _ in range(n_cycles):
        node = Node()
        cycles.append(node)
        create_cycle(node, n_links)
    return cycles

def benchamark_collection(loops, cycles, links):
    total_time = 0
    for _ in range(loops):
        gc.collect()
        all_cycles = create_gc_cycles(cycles, links)
        del all_cycles
        t0 = pyperf.perf_counter()
        collected = gc.collect()
        total_time += (pyperf.perf_counter() - t0)
        assert ((collected is None) or (collected >= (cycles * (links + 1))))
    return total_time
if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'GC link benchmark'
    runner.bench_time_func('create_gc_cycles', benchamark_collection, CYCLES, LINKS)
