import re
import pickle

class Node:
    def __init__(self,node_num,node_str):
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
                print('unknown error occurs, exit...',node)
                return
        else:
            node=node.right_child

def get_dataset(filename='dataset.dat'):
    handle=open(filename,'rb')
    dataset=pickle.load(handle)
    handle.close()
    return dataset

def post_order_traversal(node,dataset,acc):
    left_acc,right_acc=0,0
    flag_left=flag_right=False
    if node.left_child!=None:
        flag_left,left_acc=post_order_traversal(node.left_child,dataset,acc)
    if node.right_child!=None:
        flag_right,right_acc=post_order_traversal(node.right_child,dataset,acc)
    else:
        # this is the leaf node
        pruned_acc=pruning(node,dataset,acc)
        if pruned_acc>acc:
            return True,pruned_acc
    if flag_left or flag_right:
        pruned_acc=pruning(node,dataset,max(left_acc,right_acc))
        if pruned_acc>acc:
            return True,pruned_acc
    return False,acc

# Reduced-Error pruning
def pruning(node,dataset,acc):
    def find_root(node):
        temp=node
        while temp.parent!=None:
            temp=temp.parent
        return temp
    assert node!=None
    parent=node.parent
    reconstruct=''
    if parent.left_child==node:
        reconstruct='left'
        parent.left_child=None
    elif parent.right_child==node:
        reconstruct='right'
        parent.right_child=None
    else:
        assert False,str(parent)+'\t'+str(node)
    root=find_root(parent)
    true_count=0
    for data in dataset:
        label=inference(root,data)
        if data['label']==label:
            true_count+=1
    pruned_acc=true_count/num
    if pruned_acc>acc:
        return pruned_acc
    # reconstruct the tree
    if reconstruct=='left':
        parent.left_child=node
    else:
        parent.right_child=node
    return acc

if __name__ == '__main__':
    tree=decode()
    dataset=get_dataset()
    num=len(dataset)
    true_count=0
    for data in dataset:
        label=inference(tree,data)
        if data['label']==label:
            true_count+=1
    acc=true_count/num
    best_acc=post_order_traversal(tree,dataset,acc)
    print('acc=',acc,'best_acc=',best_acc)