'''
ID3 algorithm implementation

@author:zhou honggang
@date:2017/6/3
'''

import math

def split_sets(mat,col):
    rt={}
    name_row=mat[0][:col]+mat[0][col+1:]
    for i in range(1,len(mat)):
        if mat[i][col] in rt:
            rt[mat[i][col]].append(mat[i][:col]+mat[i][col+1:])
        else:
            rt[mat[i][col]]=[name_row]
            rt[mat[i][col]].append(mat[i][:col]+mat[i][col+1:])
    return  rt

def cal_entropy(submat, col,count):
    ''''
    # return the value of entropy when data is split by col
    @todo: a more efficient way is to cal entropy and choose 
        the best feature first, then cal subsets using this feature 
    @return:
        entropy: entropy when col is chosen
        subsets: split datasets
        
    '''
    entropy = 0
    # split_sets structure: feature_val -> [negative_num,positive_num]
    # split_sets is the map from feature val to
    split_sets = dict()
    for index in range(1,len(submat)):
        if submat[index][col] in split_sets:
            split_sets[submat[index][col]][submat[index][-1]] += 1
        else:
            # note that count is a list whose element is zero
            split_sets[submat[index][col]] = count.copy()
            split_sets[submat[index][col]][submat[index][-1]] += 1
    for key in split_sets:
        val = split_sets[key]
        sum_val=sum(val)
        for i in val:
            p = i / sum_val
            if p!=0:
                entropy += -p * math.log(p)
    return entropy

def build(matrix):
    ''' build decision tree
    
    :param matrix: training set with the last column being label and the first
        row being names of features
    :return: tree
    '''
    # get all possible labels
    label_vals=list(set([matrix[i][-1] for i in range(1,len(matrix))]))
    # check whether should stop building tree
    # case 1: there is no features left
    # WARN: here I assume there are only two labels
    # count=[0]*len(label_vals)
    count=[0,0]
    if len(matrix[0])==1:
        # count 1 and 0
        num=len(matrix)
        for i in range(1,num):
            count[matrix[i][-1]]+=1
        return count
    # case 2: there is no examples in current dataset
    if len(matrix) == 1:
        return count
    # case 3: all the labels are the same
    cur_label=matrix[1][-1]
    for i in range(2,len(matrix)):
        if cur_label!=matrix[i][-1]:
            break
    else:
        count[cur_label]=len(matrix)-1
        return count

    # todo: not scalable
    min_entropy=1e10
    split_ind=0
    for i in range(len(matrix[0])-1):
        entropy=cal_entropy(matrix,i,count)
        #print('feature: %s, entropy: %.2f'%(matrix[0][i],entropy))
        if min_entropy>entropy:
            min_entropy=entropy
            split_ind=i
    split_feature=matrix[0][split_ind]
    # 　Warn: subsets need to be copied from original data
    sub_sets=split_sets(matrix,split_ind)
    #print('matrix=',matrix,'choose feature:',matrix[0][split_ind],'with entropy:',min_entropy,'\nsubsets:',sub_sets)
    del matrix
    tree={'attr':split_feature}
    for partion_val in sub_sets:
        partion=sub_sets[partion_val]
        tree[partion_val]=build(partion)
    return tree

def process(mat):
    mapping={}
    mat_=mat.copy()
    col_num=len(mat[0])-1
    for i in range(1,len(mat)):
        for j in range(col_num):
            if mat[0][j] not in mapping:
                # feature name -> value index
                mapping[mat[0][j]]=mat[i][j]
                mat_[i][j]=0
            else:
                if mat[i][j] not in mapping[mat[0][j]]:
                    mapping[mat[0][j]].append(mat[i][j])
    return mapping

def read_data(filename='test_data.txt'):
    with open(filename,encoding='utf-8') as file:
        lines=file.readlines()
        mat=[line.strip().split() for line in lines]
        for i in range(1,len(mat)):
            if mat[i][-1]=='良':
                mat[i][-1]=1
            else:
                mat[i][-1]=0
        #process(mat)
    return mat

def beautiful_print(tree,layer=''):
    if tree.__class__ is dict:
        print(layer+'#'+str(tree['attr']))
        for key in tree:
            if key is not 'attr':
                val=tree[key]
                if val.__class__ is dict:
                    print(layer+'"'+str(key)+'"'+' -> \n',end='')
                    beautiful_print(val,layer+' '*len('"'+str(key)+'"'+' -> \n'))
                else:
                    print(layer+'"' + str(key) + '"' + ' -> ', end='')
                    # todo: not scalable
                    max=0
                    for i in range(len(val)):
                        if val[i]>max:
                            ind=i
                    if ind==0:
                        name="N"
                    else:
                        name='Y'
                    print(name)

if __name__ == '__main__':
    # matrix format: [names,,,label;features,,,label;;;]
    matrix=read_data()
    #print(matrix)
    tree=build(matrix)
    # pretty print
    beautiful_print(tree)