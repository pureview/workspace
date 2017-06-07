import re

def decode(filename='tree.dot'):
    nodes={}
    for line in open(filename):
        if re.match(r'[0-9]+ -> [0-9]',line.strip()):
            # edge
            parent,child=re.search(r'([0-9]+) -> ([0-9]+)').groups()
            nodes[int(parent)].child=nodes[int(child)]
            nodes[int(child)].parent=nodes[int(parent)]
        elif re.match(r'[0-9]+ \[[^\]]+\]',line.strip()):
            # node
            num, desc=re.search(r'([0-9]+) \[([\s\S]+)\]',line).groups()
            num=int(num)
            nodes[num]=Node(num,desc)

class Node:
    def __init__(self,node_num,node_str):
        self.node_num=node_num
        label=None
        eval(node_str)
        assert label!=None
        key_vals=label.strip().split('\n')
        if key_vals==4:
            self.name,self.gini,self.samples,self.value=key_vals
        elif key_vals==3:
            self.gini, self.samples, self.value = key_vals
            self.name=None
