import os
import pickle

'''
@author: zhou honggang
@diary(2017/5/23):
    1. duration is millisecond
    2. ``(该node straggler的平均read bytes/read record) /stage平均水平`` 未计算
    3. node 所有数据未归一化
@bug:
    1. straggler对象应该是原始对象√
    2. node节点统计量错误
    （2017/5/24)
@todo:
    (2017/5/24)
    1. 分析硬件信息
@note:
    (2017/5/24)
    1. 硬件相对时间比spark日志开始时间早2s
    2.cpu使用率用mpstat的第二列数据，
      IO使用率用iostat最后一列数据，
      网络使用率用sar的（第三列加第四列）/1000
    
'''

class Task(object):
    def __init__(self,task_id):
        self.task_id=task_id

def feed(filename='data/log'):
    # tasks: task_id -> task
    tasks={}
    stages={}
    # this param is to prevent multiple `SparkListenerApplicationStart`
    application_start_flag=True
    with open(filename) as file:
        while True:
            line=file.readline()
            # delete improper '\'
            line=line.replace('\"','"')
            # change `false` to 'False', 'true' to 'True'
            line=line.replace('false','False')
            line=line.replace('true','True')

            if not line:
                break
            try:
                event=eval(line)
            except:
                print('event resolution error,event:\n\t',line)
                continue
            if event['Event']=='SparkListenerTaskEnd':
                tasks[event['Task Info']['Task ID']]=event
            elif event['Event']=='SparkListenerStageCompleted':
                stages[event['Stage Info']['Stage ID']]=event
            elif event['Event']=='SparkListenerApplicationStart' and application_start_flag:
                application_start_flag=False
                # note that start_time_stamp is int variable
                start_time_stamp=event['Timestamp']
    print('log analysis finished!\n\tfind %d tasks, %d stages, application started at %d'%(
          len(tasks),len(stages),start_time_stamp))
    return start_time_stamp,tasks, stages

def catch_exception(expression,default=0):
    try:
        return eval(expression)
    except:
        return default

def load_dicts(dump_file_name='saved_raw_features'):
    if os.path.exists(dump_file_name):
        with open(dump_file_name,'rb') as dump_file:
            return pickle.load(dump_file)
    else:
        start_time_stamp,tasks,stages=feed()
        with open(dump_file_name,'wb') as dump_file:
            pickle.dump((start_time_stamp,tasks,stages),dump_file)
        return start_time_stamp,tasks,stages
def analysis_features(tasks,stages):
    def cal_stage_data_read(tasks,stages):
        for stage_id in stages.keys():
            stages[stage_id]['bytes_read']=0
            stages[stage_id]['records_read']=0
            stages[stage_id]['shuffle_read']=0
            stages[stage_id]['shuffle_records']=0
            stages[stage_id]['tasks']=[]
            stages[stage_id]['bytes_per_record_sum']=0
            stages[stage_id]['write_bytes_per_record_sum'] = 0
            stages[stage_id]['bytes_wrote']=0
            stages[stage_id]['records_wrote']=0
            stages[stage_id]['partition']=stages[stage_id]['Stage Info']['RDD Info'][0]['Number of Partitions']
            stages[stage_id]['duration']=stages[stage_id]['Stage Info']['Completion Time']- \
                stages[stage_id]['Stage Info']['Submission Time']
            stages[stage_id]['remote_fetch']=0
        for task_id in tasks:
            task=tasks[task_id]
            # get task ids of one stage
            stages[task['Stage ID']]['tasks'].append(task_id)
            if 'Input Metrics' in task['Task Metrics'].keys():
                stages[task['Stage ID']]['bytes_read']+=task['Task Metrics']['Input Metrics']['Bytes Read']
                stages[task['Stage ID']]['records_read']+=task['Task Metrics']['Input Metrics']['Records Read']
            if 'Shuffle Read Metrics' in task['Task Metrics'].keys():
                stages[task['Stage ID']]['shuffle_read']+=task['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Read']
                stages[task['Stage ID']]['shuffle_records']+=task['Task Metrics']['Shuffle Read Metrics']['Total Records Read']
                try:
                    stages[task['Stage ID']]['bytes_per_record_sum']+=task['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Read']/ \
                        task['Task Metrics']['Shuffle Read Metrics']['Total Records Read']
                except:
                    pass
            if 'Shuffle Write Metrics' in task['Task Metrics'].keys():
                stages[task['Stage ID']]['bytes_wrote']+=task['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written']
                stages[task['Stage ID']]['records_wrote']+=task['Task Metrics']['Shuffle Write Metrics']['Shuffle Records Written']
                stages[task['Stage ID']]['write_bytes_per_record_sum'] += task['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written'] / \
                                                                    task['Task Metrics']['Shuffle Write Metrics']['Shuffle Records Written']
    def find_straggler(tasks,stages,threshold=0.7):
        # straggler->task_duration/stage_duration>threshold
        stragglers={}
        for stage_id in stages:
            stage=stages[stage_id]
            stage_duration=stage['duration']
            for task_id in stage['tasks']:
                task_duration=task['Task Info']['Finish Time']-task['Task Info']['Launch Time']
                if task_duration/stage_duration>=threshold:
                    stragglers[task_id]=tasks[task_id]
        print('find %d stragglers'%(len(stragglers)))
        for k in stragglers:
            print('straggler:',stragglers[k])
            break
        return stragglers

    def init_feature(feature):
        feature['shuffle_read'] = 0
        feature['shuffle_records'] = 0
        feature['bytes_per_record'] = 0
        feature['remote_fetch'] = 0
        feature['remote_fetch_rate'] = 0
        feature['shuffle_write'] = 0
        feature['shuffle_write_bytes'] = 0
        feature['stage_id'] = task['Stage ID']
        feature['read_from_hdfs'] = 0
        feature['data_read_method'] = 0
        feature['bytes_read'] = 0
        feature['records_read'] = 0
        feature['input_bytes/result_bytes'] = 0
        feature['shuffle_write'] = 0
        feature['shuffle_write_bytes'] = 0
        feature['remote_fetch'] = 0
        feature['remote_fetch_rate'] = 0
        feature['fetch_wait_time'] = 0
        feature['data_read_method'] = 0
        feature['input_bytes/result_bytes'] = 0
        feature['shuffle_write_records']=0
    def value2bit(value,border=1):
        if value>border:
            return 1
        return 0
    def locality(value):
        if value=='PROCESS_LOCAL':
            return 1
        if value=='NODE_LOCAL':
            return 2
        return 0
    def cal_nodes(stragglers):
        nodes={}
        for task_id in stragglers:
            task=stragglers[task_id]
            node_id=int(task['Task Metrics']['Host Name'][-1])
            if node_id in nodes.keys():
                nodes[node_id].append(task_id)
            else:
                nodes[node_id]=[task_id]
        return nodes

    features={}
    cal_stage_data_read(tasks,stages)

    for task_id in tasks:
        task=tasks[task_id]
        feature={}
        # init feature
        init_feature(feature)
        # todo: this expression is not scalable
        feature['node_id']=int(task['Task Metrics']['Host Name'][-1])
        if task['Task Type']=='ResultTask':
            feature['task_type']=1
        elif task['Task Type']=='ShuffleMapTask':
            feature['task_type']=0
        feature['task_duration']=task['Task Info']['Finish Time']-\
            task['Task Info']['Launch Time']
        # todo: note that if read_from_hdfs is 0, then other features
        # may not exist and should be set properly
        if 'Input Metrics' in task['Task Metrics'].keys():
            feature['read_from_hdfs']=1
            # Hadoop -> 1, Memory -> 0, Not Exist -> -1
            if task['Task Metrics']['Input Metrics']['Data Read Method']=='Hadoop':
                feature['data_read_method']=1
            feature['bytes_read']=task['Task Metrics']['Input Metrics']['Bytes Read']/\
                stages[task['Stage ID']]['bytes_read']
            feature['records_read']=task['Task Metrics']['Input Metrics']['Records Read']/\
                stages[task['Stage ID']]['records_read']
            if task['Task Metrics']['Input Metrics']['Bytes Read']/\
                task['Task Metrics']['Result Size']>1:
                feature['input_bytes/result_bytes']=1
        if 'Shuffle Read Metrics' in task['Task Metrics'].keys():
            feature['shuffle_read']=1
            try:
                feature['shuffle_read_bytes']=task['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Read']/\
                    stages[task['Stage ID']]['shuffle_read']
            except:
                feature['shuffle_read_bytes']=0
            try:
                feature['shuffle_records'] = task['Task Metrics']['Shuffle Read Metrics']['Total Records Read'] / \
                                          stages[task['Stage ID']]['shuffle_records']
            except:
                feature['shuffle_records'] = 0
            try:
                feature['bytes_per_record']=feature['shuffle_read_bytes']/feature['shuffle_records']/\
                                            stages[task['Stage ID']]['bytes_per_record_sum']/\
                                            len(stages[task['Stage ID']]['tasks'])
            except:
                feature['bytes_per_record']=0
            if 'Reote Blocks Fetched' in task['Task Metrics']['Shuffle Read Metrics'].keys():
                feature['remote_fetch'] = 1
                # todo: maybe errors
                feature['remote_fetch_rate']=task['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Fetched']/\
                    feature['bytes_read']
                feature['fetch_wait_time']=task['Task Metrics']['Shuffle Read Metrics']['Fetch Wait Time']

        if 'Shuffle Write Metrics' in task['Task Metrics'].keys():
            feature['shuffle_write']=1
            feature['shuffle_write_bytes']=task['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written']/\
                stages[task['Stage ID']]['bytes_wrote']
            feature['shuffle_write_records'] = task['Task Metrics']['Shuffle Write Metrics']['Shuffle Records Written'] / \
                stages[task['Stage ID']]['records_wrote']
            try:
                feature['write_bytes_per_record']=feature['shuffle_write_bytes']/feature['shuffle_write_records']/\
                                            stages[task['Stage ID']]['write_bytes_per_record_sum']/\
                                            len(stages[task['Stage ID']]['tasks'])
            except:
                feature['write_bytes_per_record']=0
        try:
            feature['write_bytes/read_bytes']=value2bit(feature['shuffle_write_bytes']/feature['bytes_read'])
        except:
            feature['write_bytes/read_bytes']=0
        feature['locality']=locality(task['Task Info']['Locality'])
        try:
            feature['deserialize']=task['Task Metrics']['Executor Deserialize Time']/feature['task_duration']
            feature['executor_run_ime']=task['Task Metrics']['Executor Run Time']/feature['task_duration']
            feature['JVM_time']=task['Task Metrics']['JVM GC Time']/feature['task_duration']
            feature['serialize']=task['Task Metrics']['Result Serialization Time']/feature['task_duration']
            feature['memory_bytes_spilled']=task['Task Metrics']['Memory Bytes Spilled']/feature['bytes_read']
            feature['disk_bytes_spilled']=task['Task Metrics']['Disk Bytes Spilled']/feature['bytes_read']
        except:
            pass
        features[task_id]=feature
    node_features={}
    '''
    # sum variables between stages 
    all_bytes=0
    all_records=0
    for _,stage in stages:
        all_bytes+=stage['bytes_read']
        all_records+=stage['records_read']
    '''
    stragglers=find_straggler(tasks,stages)
    nodes=cal_nodes(stragglers)
    for node_id in nodes:
        ids=nodes[node_id]
        node_feature={}
        # mean input bytes of all stragglers in a node
        input_bytes=0
        input_records=0
        remote_fetch=0
        fetch_wait_time=0
        bytes_write=0
        records_write=0
        locality=0
        deserialize=0
        executor_run_ime =0
        JVM_time =0
        serialize =0
        memory_bytes_spilled=0
        disk_bytes_spilled =0
        for id in ids:
            try:
                input_bytes+=tasks[id]['Task Metrics']['Input Metrics']['Bytes Read']
            except:
                pass
                # print('----debug------')
                # print('input_bytes: ',tasks[id])
                # print('----debug------')
            try:
                input_records+=tasks[id]['Task Metrics']['Input Metrics']['Records Read']
            except:
                pass
            try:
                remote_fetch+=tasks[id]['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Fetched']
            except:
                pass
            try:
                fetch_wait_time+=tasks[id]['Task Metrics']['Shuffle Read Metrics']['Fetch Wait Time']
            except:
                pass
            try:
                bytes_write+=tasks[id]['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written']
            except:
                pass
            try:
                records_write+=tasks[id]['Task Metrics']['Shuffle Write Metrics']['Shuffle Records Written']
            except:
                pass
            try:
                locality+=locality(tasks[id]['Task Info']['Locality'])
            except:
                pass
            try:
                deserialize+= task['Task Metrics']['Executor Deserialize Time'] / feature['task_duration']
            except:
                pass
            try:
                executor_run_ime+=task['Task Metrics']['Executor Run Time'] / feature['task_duration']
            except:
                pass
            try:
                JVM_time += task['Task Metrics']['JVM GC Time'] / feature['task_duration']
            except:
                pass
            try:
                serialize += task['Task Metrics']['Result Serialization Time'] / feature['task_duration']
            except:
                pass
            try:
                memory_bytes_spilled += task['Task Metrics']['Memory Bytes Spilled'] / feature['bytes_read']
            except:
                pass
            try:
                disk_bytes_spilled += task['Task Metrics']['Disk Bytes Spilled'] / feature['bytes_read']
            except:
                pass
        node_feature['input_bytes']=input_bytes
        node_feature['input_records']=input_records
        node_feature['remote_fetch']=remote_fetch
        node_feature['fetch_wait_time']=fetch_wait_time
        node_feature['bytes_write']=bytes_write
        node_feature['records_write']=records_write
        node_feature['locality']=locality
        node_feature['deserialize']=deserialize
        node_feature['executor_run_ime']=executor_run_ime
        node_feature['JVM_time']=JVM_time
        node_feature['serialize']=serialize
        node_feature['memory_bytes_spilled']=memory_bytes_spilled
        node_feature['disk_bytes_spilled']=disk_bytes_spilled
        node_features[node_id]=node_feature
    return features,node_features

def read_hardware_log(filename,timestamp=0,features=[]):
    with open(filename) as log:
        rt=[]
        while True:
            line=log.readline()
            if not line:
                break
            values=[int(value) for value in line.split()]
            rt.append([values[index] for index in [timestamp,*features]])
    return rt

def expand_tasks(tasks,features,name):
    pass

def wraper(tasks):
    # get iostat
    io_files=['data/out/iostat_out_slave1','data/out/iostat_out_slave2',
              'data/out/iostat_out_slave3','data/out/iostat_out_slave4',
              'data/out/iostat_out_slave5']
    # io_features: [slave_id [[timestamp,value] ... ] ...]
    io_features=[]
    for file_id in range(len(io_files)):
        io_features.append(read_hardware_log(io_files[file_id],features=[-1]))
    expand_tasks(tasks,io_features,'io')

if __name__ == '__main__':
    _,tasks,stages=load_dicts()
    features,node_features=analysis_features(tasks,stages)
    for task_id in features:
        feature=features[task_id]
        for k in feature:
            v=feature[k]
            print(k,'->',v)
        break
    print('\nnode features:\n')
    for node_id in node_features:
        node_feature=node_features[node_id]
        for k in node_feature:
            v=node_feature[k]
            print('{} -> {}:'.format(k,v))
        #print()
        break
    # load_dicts()