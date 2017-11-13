import matplotlib.pyplot as plt
from IPython import embed

def draw_anomaly_with_tasks(package):
    ''' Draw package.
    Args:
        package: key->val format, possible keys are `cpu`, `net`, 'io`, `anomaly`, ` tasks`
    '''
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('Timeline')
    ax1.set_ylabel('Feature Scale')
    ax1.set_ylim(0,1.2)
    color={'cpu':'b','io':'g','net':'c'}
    keys=package.keys()
    for key in sorted(keys):
        obj=package[key]
        if key in ['cpu','net','io']:
            plot_x=[]
            plot_y=[]
            # we need to normalize these features
            max_y=max(obj,key=lambda x:x[1])
            if max_y[1]==0: max_y[1]=1
            for x,y in obj:
                plot_x.append(float(x))
                plot_y.append(float(y/max_y[1]))
            # cpu, net, io is wrapped in the same format
            # revert format
            ax1.plot(plot_x,plot_y,color[key],label=key)
        if key=='anomaly':
            if len(obj)==0:
                continue
            max_len=int(obj[-1][2])
            # draw rectangle
            for name,start,end in obj:
                start=int(start)
                end=int(end)
                ax1.plot([float(start),float(end)],[1.1,1.1],'k-.')
                ax1.annotate(name,xy=((start+end)/2,1.1),xytext=(start,1.12))
        if key=='tasks':
            ax2=ax1.twinx()
            #ax2.set_ylim(0,4)
            ax2.set_ylabel('Straggler Scale')
            #ax2=ax1
            for start,end,avg,root in obj:
                #embed()
                #embed()
                #scale=min(scale,100)/100
                #avg=min(avg,)
                ax2.plot([float(start),float(end)],[avg,avg],'k-',linewidth=2)
                if 'unkown' in str(root): print(root);continue
                ax2.annotate(str(root),xy=((start+end)/2,avg),xytext=(start,avg))
                #embed()
            ax2.legend()
    ax1.legend(loc=3)
    #plt.ylim(0,4)
    plt.show()

def draw_distribution(data,bucket=10,):
    ''' Draw different data occurrence.
    Args:
        data: list format, [x1, x2, ...]
        bucket: divide value into these buckets
    '''
    max_val=max(data)
    bucket_size=(max_val+0.1)/bucket
    plot_y=[0]*bucket
    bins=[]
    for i in range(bucket):
        bins.append(i*bucket_size)
    plt.hist(data,bins=bins)
    plt.xlabel('task duration/s')
    plt.ylabel('task count')
    plt.show()

def draw_errer_bar():
    data=dict()
    # read info
    name=''
    with open('info') as f:
        for line in f:
            if line.strip()=='': continue
            if line.strip().startswith('#'):
                name=line.split('=')[1].strip()
                data[name]=[]
            else:
                data[name].append(float(line))
    # calculate
    #print('data:',data)
    names=data.keys()
    names=['baseline','cpu','io','net','cpu+io+net']
    mean=[]
    err=[[],[]]
    ind=list(range(len(data)))
    for name in names:
        mean.append(sum(data[name])/len(data[name]))
        err[0].append(-min(data[name])+mean[-1])
        err[1].append(max(data[name])-mean[-1])
    # ready to plot
    plt.style.use('seaborn-whitegrid')
    plt.errorbar(ind,mean,yerr=err,marker='o',fmt='.k--')
    plt.xticks(ind,names)
    plt.ylabel('duration/s')
    #print('err:',err)
    #print('mean:',mean)
    plt.show()

if __name__ == '__main__':
    draw_errer_bar()