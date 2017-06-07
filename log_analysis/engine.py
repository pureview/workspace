import os
import pickle
import sys
# sys.path.append("/home/tsee/logtest/workspace-master")
import log_analysis.cluster_tool
import matplotlib.pyplot as plt
import log_analysis.plot_mat
import log_analysis.decision_tree
import sklearn

'''
@author: zhou honggang
@diary:
    (2017/5/23)
    1. duration is millisecond
    2. ``(该node straggler的平均read bytes/read record) /stage平均水平`` 未计算
    3. node 所有数据未归一化
    （2017/5/30)
    1. modify wraper() to get CPU and IO info from mpstat
    2. modify feed() to set start timestamp to job start time
    3. set file name through configure
    4. let analysis start all over every time
    5. get all features together and deal them with decision tree
@bug:
    1. straggler对象应该是原始对象√
    2. node节点统计量错误
    （2017/5/24)
@todo:
    (2017/5/24)
    1. 分析硬件信息√
@note:
    (2017/5/24)
    1. 硬件相对时间比spark日志开始时间早2s
    2.cpu使用率用mpstat的第二列数据，
      IO使用率用iostat最后一列数据，
      网络使用率用sar的（第三列加第四列）/1000
@cluster:
    (2017/5/25)
    1. 分成3类
     centers: [[  5.12499083e-01   9.76179435e-01   6.53056195e-04]
        [  1.06931896e+00   1.01326743e+00   4.63918748e+00]
        [  2.43434292e+00   1.05895691e+00   1.32923629e-03]]
     inertia_: 1670.95733984
'''
# ============== CONFIGURE ==================
DELAY = 12
LOG_DIR = 'data/bayes-1000/'


def feed(filename='data/log'):
    # tasks: task_id -> task
    tasks = {}
    stages = {}
    # this param is to prevent multiple `SparkListenerApplicationStart`
    application_start_flag = True
    with open(filename) as file:
        while True:
            line = file.readline()
            # delete improper '\'
            line = line.replace('\"', '"')
            # change `false` to 'False', 'true' to 'True'
            line = line.replace('false', 'False')
            line = line.replace('true', 'True')

            if not line:
                break
            try:
                event = eval(line)
            except:
                print('event resolution error,event:\n\t', line)
                continue
            if event['Event'] == 'SparkListenerTaskEnd':
                tasks[event['Task Info']['Task ID']] = event
            elif event['Event'] == 'SparkListenerStageCompleted':
                stages[event['Stage Info']['Stage ID']] = event
            elif event['Event'] == 'SparkListenerJobStart' and application_start_flag:
                application_start_flag = False
                # note that start_time_stamp is int variable
                start_time_stamp = event['Submission Time']
    print('log analysis finished!\n\tfind %d tasks, %d stages, application started at %d' % (
        len(tasks), len(stages), start_time_stamp))
    return start_time_stamp, tasks, stages

def catch_exception(expression, default=0):
    try:
        return eval(expression)
    except:
        return default

def load_dicts(dump_file_name='saved_raw_features'):
    for i in os.listdir(LOG_DIR):
        if os.path.isfile(LOG_DIR + i):
            log_file = LOG_DIR + i
    start_time_stamp, tasks, stages = feed(log_file)
    return start_time_stamp, tasks, stages
    # if os.path.exists(dump_file_name):
    #     with open(dump_file_name,'rb') as dump_file:
    #         return pickle.load(dump_file)
    # else:
    #     for i in os.listdir(LOG_DIR):
    #         if os.path.isfile(LOG_DIR+i):
    #             log_file=LOG_DIR+i
    #     start_time_stamp,tasks,stages=feed(log_file)
    #     with open(dump_file_name,'wb') as dump_file:
    #         pickle.dump((start_time_stamp,tasks,stages),dump_file)
    #     return start_time_stamp,tasks,stages


def analysis_features(tasks, stages):
    def cal_stage_data_read(tasks, stages):
        for stage_id in stages.keys():
            stages[stage_id]['task_duration_sum'] = 0
            stages[stage_id]['bytes_read'] = 0
            stages[stage_id]['records_read'] = 0
            stages[stage_id]['shuffle_read'] = 0
            stages[stage_id]['shuffle_records'] = 0
            stages[stage_id]['tasks'] = []
            stages[stage_id]['bytes_per_record_sum'] = 0
            stages[stage_id]['write_bytes_per_record_sum'] = 0
            stages[stage_id]['bytes_wrote'] = 0
            stages[stage_id]['records_wrote'] = 0
            stages[stage_id]['partition'] = stages[stage_id]['Stage Info']['RDD Info'][0]['Number of Partitions']
            stages[stage_id]['duration'] = stages[stage_id]['Stage Info']['Completion Time'] - \
                                           stages[stage_id]['Stage Info']['Submission Time']
            stages[stage_id]['remote_fetch'] = 0
        for task_id in tasks:
            task = tasks[task_id]
            # get task ids of one stage
            stages[task['Stage ID']]['tasks'].append(task_id)
            stages[task['Stage ID']]['task_duration_sum'] += task['Task Info']['Finish Time'] - task['Task Info'][
                'Launch Time']
            if 'Input Metrics' in task['Task Metrics'].keys():
                stages[task['Stage ID']]['bytes_read'] += task['Task Metrics']['Input Metrics']['Bytes Read']
                stages[task['Stage ID']]['records_read'] += task['Task Metrics']['Input Metrics']['Records Read']
            if 'Shuffle Read Metrics' in task['Task Metrics'].keys():
                stages[task['Stage ID']]['shuffle_read'] += task['Task Metrics']['Shuffle Read Metrics'][
                    'Remote Bytes Read']
                stages[task['Stage ID']]['shuffle_records'] += task['Task Metrics']['Shuffle Read Metrics'][
                    'Total Records Read']
                try:
                    stages[task['Stage ID']]['bytes_per_record_sum'] += task['Task Metrics']['Shuffle Read Metrics'][
                                                                            'Remote Bytes Read'] / \
                                                                        task['Task Metrics']['Shuffle Read Metrics'][
                                                                            'Total Records Read']
                except:
                    pass
            if 'Shuffle Write Metrics' in task['Task Metrics'].keys():
                stages[task['Stage ID']]['bytes_wrote'] += task['Task Metrics']['Shuffle Write Metrics'][
                    'Shuffle Bytes Written']
                stages[task['Stage ID']]['records_wrote'] += task['Task Metrics']['Shuffle Write Metrics'][
                    'Shuffle Records Written']
                stages[task['Stage ID']]['write_bytes_per_record_sum'] += task['Task Metrics']['Shuffle Write Metrics'][
                                                                              'Shuffle Bytes Written'] / \
                                                                          task['Task Metrics']['Shuffle Write Metrics'][
                                                                              'Shuffle Records Written']
    # def find_straggler(tasks,stages,features,threshold=0.7):
    def find_straggler(tasks, stages, features, threshold=1.5):
        # straggler->task_duration/stage_duration>threshold
        stragglers = {}
        for stage_id in stages:
            stage = stages[stage_id]
            # stage_duration=stage['duration']
            stage_avg_duration = stage['task_duration_sum'] / len(stage['tasks'])
            for task_id in stage['tasks']:
                task_duration = task['Task Info']['Finish Time'] - task['Task Info']['Launch Time']
                # if task_duration/stage_duration>=threshold:
                if task_duration / stage_avg_duration >= threshold:
                    stragglers[task_id] = tasks[task_id]
                    features[task_id]['straggler'] = 1
        print('find %d stragglers' % (len(stragglers)))
        # for k in stragglers:
        #     print('straggler:',stragglers[k])
        #     break
        return stragglers

    def init_feature(feature):
        feature['shuffle_read'] = 0
        feature['shuffle_records'] = 0
        feature['bytes_per_record'] = 0
        feature['remote_fetch'] = 0
        feature['remote_fetch_rate'] = 0
        feature['shuffle_write'] = 0
        feature['shuffle_write_bytes'] = 0
        feature['stage_id'] = 0
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
        feature['shuffle_write_records'] = 0
        feature['straggler'] = 0
        feature['node_id'] = 0
        feature['task_type'] = 0
        feature['task_duration'] = 0
        feature['shuffle_read_bytes'] = 0
        feature['write_bytes_per_record'] = 0
        feature['write_bytes/read_bytes'] = 0
        feature['deserialize'] = 0
        feature['executor_run_ime'] = 0
        feature['JVM_time'] = 0
        feature['serialize'] = 0
        feature['memory_bytes_spilled'] = 0
        feature['disk_bytes_spilled'] = 0
        feature['locality'] = 0

    def value2bit(value, border=1):
        if value > border:
            return 1
        return 0

    def locality(value):
        if value == 'PROCESS_LOCAL':
            return 1
        if value == 'NODE_LOCAL':
            return 2
        return 0

    def cal_nodes(stragglers):
        nodes = {}
        for task_id in stragglers:
            task = stragglers[task_id]
            node_id = int(task['Task Metrics']['Host Name'][-1])
            if node_id in nodes.keys():
                nodes[node_id].append(task_id)
            else:
                nodes[node_id] = [task_id]
        return nodes

    features = {}
    cal_stage_data_read(tasks, stages)
    for task_id in tasks:
        task = tasks[task_id]
        feature = {}
        # init feature
        init_feature(feature)
        # todo: this expression is not scalable
        feature['node_id'] = int(task['Task Metrics']['Host Name'][-1])
        if task['Task Type'] == 'ResultTask':
            feature['task_type'] = 1
        elif task['Task Type'] == 'ShuffleMapTask':
            feature['task_type'] = 0
        feature['task_duration'] = task['Task Info']['Finish Time'] - \
                                   task['Task Info']['Launch Time']
        # todo: note that if read_from_hdfs is 0, then other features
        # may not exist and should be set properly
        if 'Input Metrics' in task['Task Metrics'].keys():
            feature['read_from_hdfs'] = 1
            # Hadoop -> 1, Memory -> 0, Not Exist -> -1
            if task['Task Metrics']['Input Metrics']['Data Read Method'] == 'Hadoop':
                feature['data_read_method'] = 1
            feature['bytes_read'] = len(stages[task['Stage ID']]['tasks']) * task['Task Metrics']['Input Metrics'][
                'Bytes Read'] / \
                                    stages[task['Stage ID']]['bytes_read']
            feature['records_read'] = len(stages[task['Stage ID']]['tasks']) * task['Task Metrics']['Input Metrics'][
                'Records Read'] / \
                                      stages[task['Stage ID']]['records_read']
            if task['Task Metrics']['Result Size'] > 0 and task['Task Metrics']['Input Metrics']['Bytes Read'] / \
                    task['Task Metrics']['Result Size'] > 1:
                feature['input_bytes/result_bytes'] = 1
        if 'Shuffle Read Metrics' in task['Task Metrics'].keys():
            feature['shuffle_read'] = 1
            try:
                feature['shuffle_read_bytes'] = len(stages[task['Stage ID']]['tasks']) * (
                task['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Read'] + \
                task['Task Metrics']['Shuffle Read Metrics']['Local Bytes Read']) / \
                                                stages[task['Stage ID']]['shuffle_read']
            except:
                feature['shuffle_read_bytes'] = 0
            try:
                feature['shuffle_records'] = len(stages[task['Stage ID']]['tasks']) * \
                                             task['Task Metrics']['Shuffle Read Metrics']['Total Records Read'] / \
                                             stages[task['Stage ID']]['shuffle_records']
            except:
                feature['shuffle_records'] = 0
            try:
                feature['bytes_per_record'] = feature['shuffle_read_bytes'] / feature['shuffle_records'] / \
                                              stages[task['Stage ID']]['bytes_per_record_sum'] / \
                                              len(stages[task['Stage ID']]['tasks'])
            except:
                feature['bytes_per_record'] = 0
            if 'Reote Blocks Fetched' in task['Task Metrics']['Shuffle Read Metrics'].keys():
                feature['remote_fetch'] = 1
                # todo: maybe errors
                feature['remote_fetch_rate'] = task['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Fetched'] / \
                                               feature['bytes_read']
                feature['fetch_wait_time'] = task['Task Metrics']['Shuffle Read Metrics']['Fetch Wait Time'] / \
                                             feature['task_duration']

        if 'Shuffle Write Metrics' in task['Task Metrics'].keys():
            feature['shuffle_write'] = 1
            feature['shuffle_write_bytes'] = task['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written'] / \
                                             stages[task['Stage ID']]['bytes_wrote']
            feature['shuffle_write_records'] = task['Task Metrics']['Shuffle Write Metrics'][
                                                   'Shuffle Records Written'] / \
                                               stages[task['Stage ID']]['records_wrote']
            try:
                feature['write_bytes_per_record'] = feature['shuffle_write_bytes'] / feature['shuffle_write_records'] / \
                                                    stages[task['Stage ID']]['write_bytes_per_record_sum'] / \
                                                    len(stages[task['Stage ID']]['tasks'])
            except:
                feature['write_bytes_per_record'] = 0
        try:
            feature['write_bytes/read_bytes'] = value2bit(feature['shuffle_write_bytes'] / feature['bytes_read'])
        except:
            feature['write_bytes/read_bytes'] = 0
        feature['locality'] = locality(task['Task Info']['Locality'])
        try:
            feature['deserialize'] = task['Task Metrics']['Executor Deserialize Time'] / feature['task_duration']
            feature['executor_run_ime'] = task['Task Metrics']['Executor Run Time'] / feature['task_duration']
            feature['JVM_time'] = task['Task Metrics']['JVM GC Time'] / feature['task_duration']
            feature['serialize'] = task['Task Metrics']['Result Serialization Time'] / feature['task_duration']
            feature['memory_bytes_spilled'] = task['Task Metrics']['Memory Bytes Spilled'] / feature['bytes_read']
            feature['disk_bytes_spilled'] = task['Task Metrics']['Disk Bytes Spilled'] / feature['bytes_read']
        except:
            pass
        features[task_id] = feature
    node_features = {}
    '''
    # sum variables between stages 
    all_bytes=0
    all_records=0
    for _,stage in stages:
        all_bytes+=stage['bytes_read']
        all_records+=stage['records_read']
    '''
    stragglers = find_straggler(tasks, stages, features)
    nodes = cal_nodes(stragglers)
    for node_id in nodes:
        ids = nodes[node_id]
        node_feature = {}
        # mean input bytes of all stragglers in a node
        input_bytes = 0
        input_records = 0
        remote_fetch = 0
        fetch_wait_time = 0
        bytes_write = 0
        records_write = 0
        locality = 0
        deserialize = 0
        executor_run_ime = 0
        JVM_time = 0
        serialize = 0
        memory_bytes_spilled = 0
        disk_bytes_spilled = 0
        for id in ids:
            try:
                input_bytes += tasks[id]['Task Metrics']['Input Metrics']['Bytes Read']
            except:
                pass
                # print('----debug------')
                # print('input_bytes: ',tasks[id])
                # print('----debug------')
            try:
                input_records += tasks[id]['Task Metrics']['Input Metrics']['Records Read']
            except:
                pass
            try:
                remote_fetch += tasks[id]['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Fetched']
            except:
                pass
            try:
                fetch_wait_time += tasks[id]['Task Metrics']['Shuffle Read Metrics']['Fetch Wait Time']
            except:
                pass
            try:
                bytes_write += tasks[id]['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written']
            except:
                pass
            try:
                records_write += tasks[id]['Task Metrics']['Shuffle Write Metrics']['Shuffle Records Written']
            except:
                pass
            try:
                locality += locality(tasks[id]['Task Info']['Locality'])
            except:
                pass
            try:
                deserialize += task['Task Metrics']['Executor Deserialize Time'] / feature['task_duration']
            except:
                pass
            try:
                executor_run_ime += task['Task Metrics']['Executor Run Time'] / feature['task_duration']
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
        node_feature['input_bytes'] = input_bytes
        node_feature['input_records'] = input_records
        node_feature['remote_fetch'] = remote_fetch
        node_feature['fetch_wait_time'] = fetch_wait_time
        node_feature['bytes_write'] = bytes_write
        node_feature['records_write'] = records_write
        node_feature['locality'] = locality
        node_feature['deserialize'] = deserialize
        node_feature['executor_run_ime'] = executor_run_ime
        node_feature['JVM_time'] = JVM_time
        node_feature['serialize'] = serialize
        node_feature['memory_bytes_spilled'] = memory_bytes_spilled
        node_feature['disk_bytes_spilled'] = disk_bytes_spilled
        node_features[node_id] = node_feature
    return features, node_features, stragglers


def read_hardware_log(filename, timestamp=0, features=[]):
    with open(filename) as log:
        rt = []
        while True:
            line = log.readline()
            if not line:
                break
            values = [float(value) for value in line.split()]
            rt.append([values[index] for index in [timestamp, *features]])
    assert len(rt[0]) >= 2, 'hardware feature fault: ' + str(rt[0])
    return rt


def expand_tasks(tasks, features, name, timestamp, extracted_features, delay=2000):
    # Note: features is refered to hardware features
    for task_id in tasks:
        task = tasks[task_id]
        task_start = task['Task Info']['Launch Time']
        task_finish = task['Task Info']['Finish Time']
        slave_index = int(task['Task Info']['Host'][-1]) - 1
        temp_sum = 0
        # NOTE: here I include start point and end point
        start_id = int((task_start - timestamp) / 1000 - DELAY)
        end_id = int((task_finish - timestamp) / 1000 - DELAY)
        assert end_id >= start_id
        for i in range(start_id, end_id + 1):
            try:
                temp_sum += features[slave_index][i][1]
            except:
                continue
        avg = temp_sum / (end_id - start_id + 1)
        task[name] = avg
        extracted_features[task_id][name] = avg


def wraper(tasks, start_time, extracted_features):
    # WARN: tasks need to be raw tasks
    # '''
    # get iostat
    io_files = [LOG_DIR + 'out/iostat_out_slave1', LOG_DIR + 'out/iostat_out_slave2',
                LOG_DIR + 'out/iostat_out_slave3', LOG_DIR + 'out/iostat_out_slave4',
                LOG_DIR + 'out/iostat_out_slave5']
    # io_features: [slave_id [[timestamp,value] ... ] ...]
    io_features = []
    for file_id in range(len(io_files)):
        io_features.append(read_hardware_log(io_files[file_id], features=[-1]))
    expand_tasks(tasks, io_features, 'io', start_time, extracted_features)
    # '''
    # cpu_features
    cpu_files = [LOG_DIR + 'out/mpstat_out_slave1', LOG_DIR + 'out/mpstat_out_slave2',
                 LOG_DIR + 'out/mpstat_out_slave3', LOG_DIR + 'out/mpstat_out_slave4',
                 LOG_DIR + 'out/mpstat_out_slave5']
    cpu_features = []
    for file_id in range(len(cpu_files)):
        cpu_features.append(read_hardware_log(cpu_files[file_id], features=[1]))
        # io_t=[]
        # cpu_t=[]
        # for item in rt:
        #     io_t.append([item[0],item[1]])
        #     cpu_t.append([item[0],item[1]])
        # io_features.append(io_t)
        # cpu_features.append(cpu_t)
    expand_tasks(tasks, cpu_features, 'cpu', start_time, extracted_features)
    # net_features
    net_files = [LOG_DIR + 'out/sar_out_slave1', LOG_DIR + 'out/sar_out_slave2',
                 LOG_DIR + 'out/sar_out_slave3', LOG_DIR + 'out/sar_out_slave4',
                 LOG_DIR + 'out/sar_out_slave5']
    net_features = []
    for file_id in range(len(net_files)):
        raw = read_hardware_log(net_files[file_id], features=[2, 3])
        net_features.append([[row[0], (row[1] + row[2]) / 1000] for row in raw])
    expand_tasks(tasks, net_features, 'net', start_time, extracted_features)


def visualize(features):
    for task_id in features:
        feature = features[task_id]
        for k in feature:
            v = feature[k]
            print(k, '->', v)
        break


def regulize(mat):
    # WARN: mat is a 2-d list
    row_num = len(mat)
    col_num = len(mat[0])
    #    sum_list=[0]*col_num
    min_list = [0] * col_num
    max_list = [0] * col_num
    flag = True
    for row in mat:
        for i in range(col_num):
            # sum_list[i]+=row[i]/row_num
            if flag:
                min_list[i] = row[i]
                max_list[i] = row[i]
            else:
                if row[i] < min_list[i]:
                    min_list[i] = row[i]
                if row[i] > max_list[i]:
                    max_list[i] = row[i]
        flag = False
    for i in range(row_num):
        for j in range(col_num):
            try:
                mat[i][j] = (mat[i][j] - min_list[j]) / (max_list[j] - min_list[j])
            except:
                mat[i][j] = 0


if __name__ == '__main__':
    # this step save cal time but hurt debug
    # if os.path.exists('stragglers.dat'):
    #     stragglers,features,tasks=pickle.load(open('straggles.dat','wb'))
    # else:
    #     start_time, tasks, stages = load_dicts()
    #     # Note: stragglers -> task_id : task
    #     features, node_features, stragglers = analysis_features(tasks, stages)
    #     wraper(tasks, start_time, features)
    #     pickle.dump([stragglers,features,tasks],open('straggles.dat','wb'))
    dataset = []
    keys = []
    labels = []
    flag_key = True
    for workload in os.listdir('data'):
        LOG_DIR = 'data/' + workload + '/'
        if 'kmeans' in workload or 'nweight' in workload:
            DELAY = 13
        else:
            DELAY = 12
        start_time, tasks, stages = load_dicts()
        # Note: stragglers -> task_id : task
        features, node_features, stragglers = analysis_features(tasks, stages)
        wraper(tasks, start_time, features)
        # get straggler hardware features
        # ----------------- Here features include everything ----------------------
        for task_id in features:
            # feature_=features[task_id].copy()
            feature_ = features[task_id]
            label = feature_.pop('straggler')
            # ----------------- Remove some features for test data 6.1 ----------------
            feature_.pop('node_id')
            feature_.pop('task_type')
            feature_.pop('task_duration')
            feature_.pop('read_from_hdfs')
            feature_.pop('records_read')
            feature_.pop('input_bytes/result_bytes')
            feature_.pop('shuffle_read')
            feature_.pop('bytes_per_record')
            feature_.pop('remote_fetch')
            feature_.pop('shuffle_write')
            feature_.pop('write_bytes_per_record')
            feature_.pop('write_bytes/read_bytes')

            labels.append(label)
            row = []
            for key in feature_:
                if flag_key:
                    keys.append(key)
                row.append(feature_[key])
            flag_key = False
            dataset.append(row)
    accuracy, precision, recall = log_analysis.decision_tree.build_tree(dataset, labels, keys)
    print('accuracy,precision,recall=', accuracy, precision, recall)
    exit()
    # clean dataset
    feature_values = {}
    for key in dataset[0][0]:
        feature_values[key] = []
        for piece in dataset:
            piece = piece[0]
            feature_values.append(piece[key])
