'''
@author: zhou honggang
@bug: 1. when pruning a tree, I did not delete abandoned 
    nodes to release memory and that might cause memory
    leakage. But this program will end in a short time so
    I think it doesn't matter.
'''

import re
import pickle
import pydotplus

class Node:
    def __init__(self,node_num,node_str):
        self.node_str=node_str
        self.node_num=node_num
        label=node_str.split('=',1)[1]
        key_vals=label.strip().split('\\n')
        if len(key_vals)==4:
            name,gini,samples,value=key_vals
            self.name=name.split('<=')[0].strip().split('"')[1]
            self.border=float(name.split('<=')[1])
            self.gini=float(gini.split('=')[1].strip())
            self.samples=int(samples.split('=')[1].strip())
            x,y=value.split('[')[1].split(']')[0].split(',')
            self.value=[int(x),int(y)]
        elif len(key_vals)==3:
            gini, samples, value = key_vals
            self.name=None
            self.border=None
            self.gini = float(gini.split('=')[1].strip())
            self.samples = int(samples.split('=')[1].strip())
            x, y = value.split('[')[1].split(']')[0].split(',')
            self.value = [int(x), int(y)]
        else:
            raise Exception('decode error! key_vals='+str(key_vals))
        assert len(self.value)==2,'value format error! '+node_str
        self.left_child=None
        self.right_child=None
        self.parent=None

    def copy(self):
        def traversal(node):
            if node.left_child!=None:
                left=traversal(node.left_child)
                right=traversal(node.right_child)
                cur=Node(node.node_num,node.node_str)
                cur.left_child=left
                cur.right_child=right
                return cur
            else:
                return Node(node.node_num,node.node_str)
        def find_your_mother(tree):
            # first order traversal
            if tree.left_child!=None:
                tree.left_child.parent=tree
                tree.right_child.parent=tree
                find_your_mother(tree.left_child)
                find_your_mother(tree.right_child)
        new=traversal(self)
        find_your_mother(new)
        return new
    def __str__(self):
        return ' gini = '+str(self.gini)+ ' samples = '+\
               str(self.samples)+' values = '+str(self.value)

def decode(filename='tree.dot'):
    nodes={}
    for line in open(filename):
        if re.match(r'[0-9]+ -> [0-9]',line.strip()):
            # edge
            parent,child=re.search(r'([0-9]+) -> ([0-9]+)',line).groups()
            if nodes[int(parent)].left_child==None:
                nodes[int(parent)].left_child=nodes[int(child)]
            else:
                nodes[int(parent)].right_child=nodes[int(child)]
            nodes[int(child)].parent=nodes[int(parent)]
        elif re.match(r'[0-9]+ \[[^\]]+\]',line.strip()):
            # node
            num, desc=re.search(r'([0-9]+) \[([\s\S]+)\]',line).groups()
            num=int(num)
            nodes[num]=Node(num,desc)
    return nodes[0]

def inference(tree,sample):
    '''
    Inference label for a sample
    Args： 
        tree: root Node object
        sample: name-value pair 
    return：
        label: predict label 
    '''
    node=tree
    while True:
        if node.left_child==None and node.right_child==None:
            # we've come to leaf node
            value=node.value
            if value[0]>value[1]:
                return 0
            else:
                return 1
        if sample[node.name]<=node.border:
            # choose left child
            if node.left_child!=None:
                node=node.left_child
            else:
                print('unknown error occurs, exit... ')
                return
        else:
            node=node.right_child

def get_dataset(filename='dataset.dat'):
    handle=open(filename,'rb')
    dataset=pickle.load(handle)
    handle.close()
    return dataset

def post_order_traversal(node,dataset,acc,sample=None,gini=None):
    '''
    post order traversal the whole tree to pruning
    :param node: Node instanse
    :param dataset: testing dataset
    :param acc: current best acc
    :param thresh: pruning threshold
    :return flag: whether this node is a leaf node
             acc: current acc
    '''
    left_acc,right_acc=0,0
    flag_left=flag_right=False
    if node.left_child!=None:
        flag_left,left_acc=post_order_traversal(node.left_child,dataset,acc)
    if node.right_child!=None:
        flag_right,right_acc=post_order_traversal(node.right_child,dataset,acc)
    else:
        # this is the leaf node， should return directly
        return True,acc
    if flag_left and flag_right:
        flag_pruned,pruned_acc=pruning(node,dataset,max(left_acc,right_acc))
        if flag_pruned:
            return True,pruned_acc
        else:
            return False,max(left_acc,right_acc)
    return False,acc

# Reduced-Error pruning
def pruning(node,dataset,acc,sample=None,gini=None):
    '''
    prune the leaf node of 
    :param node: node to be pruned
    :param dataset: test dataset
    :param acc: best acc of children
    :return: current best acc
    '''
    global pruning_count
    def find_root(node):
        temp=node
        while temp.parent!=None:
            temp=temp.parent
        return temp
    left_child=node.left_child
    right_child=node.right_child
    # try remove the children
    node.left_child=None
    node.right_child=None
    root=find_root(node)
    true_count=0
    for data in dataset:
        label=inference(root,data)
        if data['label']==label:
            true_count+=1
    pruned_acc=true_count/len(dataset)
    if pruned_acc>=acc:
        pruning_count+=1
        return True,pruned_acc
    # reconstruct the tree
    node.left_child=left_child
    node.right_child=right_child
    return False,acc

def export(tree):
    '''export tree to graphviz format
    Args:
        tree: root node of decision tree
    Return:
        out: graphviz string
    '''
    def first_order_traversal(node):
        global out,i
        out+='digraph Tree {\nnode [shape=box] ;'
        if node==None:
            return
        if node.name!=None:
            out+='\n%d [label="%s\\ngini=%.4f\\nsamples=%d\\nvalue=%s"];'%(
                i,node.name, node.gini, node.samples, str(node.value)
            )
        else:
            out += '\n%d [label="gini=%.4f\\nsamples=%d\\nvalue=%s"];' % (
                i,node.gini, node.samples, str(node.value)
            )
        # add attr i to this node
        node.i=i
        # link this node with its parent if it has one
        if node.parent!=None:
            out += '\n%d -> %d;'%(node.parent.i,i)
        # update global counter
        i+=1
        # traversal another
        first_order_traversal(node.left_child)
        first_order_traversal(node.right_child)

    global out,i
    first_order_traversal(tree)
    out+='\n}'

def mypruning_traversal(node,sample=None,gini=None):
    if node==None:
        return
    if sample!=None:
        # pruning using sample threshold
        if node.left_child and node.right_child:
            # pruning current node
            left=node.left_child
            right=node.right_child
            if left.samples<=sample and right.samples<=sample:
                node.left_child=None
                node.right_child=None
                del left,right
        mypruning_traversal(node.left_child,sample,gini)
        mypruning_traversal(node.right_child,sample,gini)
    if gini!=None:
        # pruning using gini threshold
        if node.left_child and node.right_child:
            # pruning current node
            left=node.left_child
            right=node.right_child
            if left.gini<gini and right.gini<gini:
                node.left_child=None
                node.right_child=None
                del left,right
        mypruning_traversal(node.left_child,sample,gini)
        mypruning_traversal(node.right_child,sample,gini)

def stair_test_sample(tree,dataset,thresh,stair,gini=False):
    if gini:
        # pruning using gini
        coords=[]
        i=0
        while i<thresh:
            mypruning_traversal(tree,gini=i)
            true_count=0
            for data in dataset:
                label = inference(tree, data)
                if data['label'] == label:
                    true_count += 1
            pruned_acc = true_count / len(dataset)
            coords.append([i,pruned_acc])
            # update i
            i+=stair
    else:
        # pruning using samples
        coords=[]
        i=0
        while i<thresh:
            mypruning_traversal(tree,sample=i)
            true_count = 0
            for data in dataset:
                label = inference(tree, data)
                if data['label'] == label:
                    true_count += 1
            pruned_acc = true_count / len(dataset)
            coords.append([i, pruned_acc])
            # update i
            i+=stair
    return coords

# global vars
out=''
i=0
pruning_count=0

if __name__ == '__main__':
    tree=decode()
    dataset=get_dataset()
    # stair test
    thresh=0.3
    stair=0.01
    coords=stair_test_sample(tree,dataset,thresh,stair,gini=True)
    for coord in coords:
        print('%.2f -> %.4f'%(coord[0], coord[1]))
    # num=len(dataset)
    # true_count=0
    # for data in dataset:
    #     label=inference(tree,data)
    #     if data['label']==label:
    #         true_count+=1
    # acc=true_count/num
    # best_acc=post_order_traversal(tree,dataset,acc)
    # print('acc=',acc,'best_acc=',best_acc)
    # print('pruned %d times'%(i))
    # print('pruned tree is below')
    # export(tree)
    # # dump tree to a pdf
    # graph=pydotplus.graph_from_dot_data(out)
    # graph.write_pdf('pruned_tree.pdf')
    # print("Done!")