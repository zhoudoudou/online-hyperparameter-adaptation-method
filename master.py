import random, time, queue
from copy import deepcopy, copy
from multiprocessing.managers import BaseManager
import numpy as np
# 发送任务的队列:
import csv
import tensorflow as tf
# record the best pop
#
summary_writer = tf.summary.FileWriter('./pbt')
# 在PBT的基础上，进行交叉和变异
popsize=10
task_queue = queue.Queue(maxsize = popsize)

# 接收结果的队列:
result_queue = queue.Queue(maxsize = popsize)
# 记录文件
out1 = open('ga_fitness_1.csv','a',newline='')
csv_write1 = csv.writer(out1,dialect='excel') # use for plot
out2 = open("ga_hyperpara_1.csv","a",newline='')
csv_write2 = csv.writer(out2,dialect='excel')

out3 = open('ga_fitness_2.csv','a',newline='')
csv_write3 = csv.writer(out3,dialect='excel') # use for plot
out4 = open("ga_hyperpara_2.csv","a",newline='')
csv_write4 = csv.writer(out4,dialect='excel')

out5 = open('ga_fitness_3.csv','a',newline='')
csv_write5 = csv.writer(out5,dialect='excel') # use for plot
out6 = open("ga_hyperpara_3.csv","a",newline='')
csv_write6 = csv.writer(out6,dialect='excel')

out7 = open('ga_fitness_4.csv','a',newline='')
csv_write7 = csv.writer(out7,dialect='excel') # use for plot
out8 = open("ga_hyperpara_4.csv","a",newline='')
csv_write8 = csv.writer(out8,dialect='excel')

out9 = open("ga_fitness_5.csv","a",newline='')
csv_write9 = csv.writer(out9,dialect='excel')
out10 = open("ga_hyperpara_5.csv","a",newline='')
csv_write10 = csv.writer(out10,dialect='excel')

def evolution_ga(update):
    temp=[]
    for i in range(popsize):
        temp.append(result.get())

    global_fitness = []
    global_hyper = []
    #global_weight = []
    for i in range(popsize):
        global_fitness.append(temp[i]['fitness'])
        global_hyper.append(temp[i]["hyperpara_sgd"])
        global_hyper.append(temp[i]["hyperpara_reward"])

    fitness = max(global_fitness)# acquire the max fitness
    summary = tf.Summary(value=[
        tf.Summary.Value(tag="fitness_best", simple_value=fitness)
    ])
    summary_writer.add_summary(summary, update)
    print("The global_fitness is :",global_fitness)
    print("the global_hyperpara is:",global_hyper)
    #print("the global_weight is:", global_weight)
    global_fitness = np.array(global_fitness)
    index = np.argsort(global_fitness)# return the min -> max index
    # record the fitness
    csv_write1.writerow(temp[index[popsize-1]]["episode_record"])# record the best episode
    csv_write3.writerow(temp[index[popsize-2]]["episode_record"])# record the best episode
    csv_write5.writerow(temp[index[popsize-3]]["episode_record"])# record the best episode
    csv_write7.writerow(temp[index[popsize-4]]["episode_record"])# record the best episode
    csv_write9.writerow(temp[index[popsize-5]]["episode_record"])# record the best episode
    # record the hyperparameter
    csv_write2.writerow([temp[index[popsize-1]]["hyperpara_sgd"],temp[index[popsize-1]]["hyperpara_reward"]])
    csv_write4.writerow([temp[index[popsize-2]]["hyperpara_sgd"],temp[index[popsize-2]]["hyperpara_reward"]])
    csv_write6.writerow([temp[index[popsize-3]]["hyperpara_sgd"],temp[index[popsize-3]]["hyperpara_reward"]])
    csv_write8.writerow([temp[index[popsize-4]]["hyperpara_sgd"],temp[index[popsize-4]]["hyperpara_reward"]])
    csv_write10.writerow([temp[index[popsize-5]]["hyperpara_sgd"],temp[index[popsize-5]]["hyperpara_reward"]])

    # new pbt operation
    # deepcopy the weight and hyperpara
    # 两个最差的个体用最好的个体代替
    temp[index[0]] = deepcopy(temp[index[popsize-1]])
    temp[index[1]] = deepcopy(temp[index[popsize-2]])
    # 现在temp已经是经过选择后的个体
    # tuning ent_coef, vf_coef, lr_coef
    # 需要先对tmep做一个排序
    # 对后80%进行选择交叉变异
    #
    #我们对种群进行一个处理，将适应度低的的放前面，适应度高的放后面
    temp_new = []
    for i in range(popsize):
        temp_new.append(deepcopy(temp[index[i]]))

    # 对前面8个进行交叉变异
    #crossover
    for i in range(int((popsize-2)/2)):

        if random.random() <= 0.7:
            man = temp_new[0 + 2*i]['hyperpara_sgd']+temp_new[0 + 2*i]['hyperpara_reward']
            print("man:", man)
            women = temp_new[1 + 2*i]['hyperpara_sgd']+temp_new[1 + 2*i]['hyperpara_reward']
            print("wowen:",women)
            cross_index_1 = random.randint(0, 6)
            cross_index_2 = random.randint(0, 6)
            if cross_index_2 < cross_index_1:
                cross_index_1 , cross_index_2 = cross_index_2 , cross_index_1

            print("cross_index_1",cross_index_1)
            print("cross_index_2",cross_index_2)

            temp = deepcopy(man[cross_index_1:cross_index_2 + 1])
            man[cross_index_1:cross_index_2 + 1] = deepcopy(women[cross_index_1:cross_index_2 + 1])
            women[cross_index_1:cross_index_2 + 1] = deepcopy(temp)
            print("new man:", man)
            print("new woman:",women)
            temp_new[0 + 2 * i]['hyperpara_sgd'] = deepcopy(man[0:5])
            temp_new[0 + 2 * i]['hyperpara_reward'] = deepcopy(man[5:7])
            temp_new[1 + 2 * i]['hyperpara_sgd'] = deepcopy(women[0:5])
            temp_new[1 + 2 * i]['hyperpara_reward'] = deepcopy(women[5:7])

    new_global_hyperpara = []
    for i in range(popsize):
        new_global_hyperpara.append(temp_new[i]["hyperpara_sgd"])
        new_global_hyperpara.append(temp_new[i]["hyperpara_reward"])
    print("new hyperpara by crossover:", new_global_hyperpara)

    # mulation，已个体为单位进行变异，适应度最高的两个个体不变异
    for i in range(popsize-2):
        if random.random() <= 0.25:
            # ent_coef, vf_coef,lr
            for j in range(3):
                if random.random() <= 0.5:
                    temp_new[i]['hyperpara_sgd'][j] *= 0.8
                else:
                    temp_new[i]['hyperpara_sgd'][j] *= 1.2

            # alpha
            if random.random() <= 0.5:
                temp_new[i]['hyperpara_sgd'][3] *= 0.8
            else:
                temp_new[i]['hyperpara_sgd'][3] *= 1.2
                if temp_new[i]['hyperpara_sgd'][3] >= 0.99:
                    temp_new[i]['hyperpara_sgd'][3] = 0.99
                else:
                    pass

            # nstep
            if random.random() <= 0.5:
                temp_new[i]["hyperpara_reward"][0] = deepcopy(int(temp_new[i]["hyperpara_reward"][0] * 1.2))
            else:
                temp_new[i]["hyperpara_reward"][0] = deepcopy(int(temp_new[i]["hyperpara_reward"][0] * 0.8))
                if temp_new[i]["hyperpara_reward"][0] < 5:
                    temp_new[i]["hyperpara_reward"][0] = 5
                else:
                    pass
            # gamma
            if random.random() <= 0.5:
                temp_new[i]["hyperpara_reward"][1] *= 0.8
            else:
                temp_new[i]["hyperpara_reward"][1] *= 1.2
                if temp_new[i]["hyperpara_reward"][1] >= 0.99:
                    temp_new[i]["hyperpara_reward"][1] = 0.99
                else:
                    pass

        else:
            pass

    # print new hyperpara
    new_global_hyperpara = []
    for i in range(popsize):
        new_global_hyperpara.append(temp_new[i]["hyperpara_sgd"])
        new_global_hyperpara.append(temp_new[i]["hyperpara_reward"])
    print("new hyperpara by multion:", new_global_hyperpara)

    for i in range(popsize):
        task.put(temp_new[i])


    print("the task put ok")

#def evolution()
# 从BaseManager继承的QueueManager:
class QueueManager(BaseManager):
    pass

# 把两个Queue都注册到网络上, callable参数关联了Queue对象:
QueueManager.register('master_task', callable=lambda: task_queue)
QueueManager.register('worker_result', callable=lambda: result_queue)
# 绑定端口5000, 设置验证码'abc':
manager = QueueManager(address=('', 5000), authkey=b'abc')
# 启动Queue:
manager.start()
# 获得通过网络访问的Queue对象:
task = manager.master_task()
result = manager.worker_result()

task_init=[]
for i in range(popsize):
    task.put(task_init)

for update in range(50):
    while(~result.full()):
        print("Now is the update is %d;the result num is :%d;the task num is :%d" % (update,result.qsize(),task.qsize()))
        time.sleep(2)
        if(result.full()):
            print("the result num is :%d;the task num is :%d" % (result.qsize(), task.qsize()))
            print("Now the result is full ,begin evolute")
            break
    evolution_ga(update)

manager.shutdown()
print('master exit.')


