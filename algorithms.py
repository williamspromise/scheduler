

import json
import copy
from sys import *
from math import gcd
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
from collections import defaultdict

tasks = dict()
RealTime_task = dict()
metrics = defaultdict(dict)
d = dict()
dList = []
T = []
C = []
U = []
# For gantt chart
y_axis  = []
from_x = []
to_x = []
n = 0

def analyze(data,txt):
    print(txt)
    if int(txt) == 4:
        feasible = U_factor <= 1
        return tasks,feasible
    
    feasible = True
    for i  in range(len(data)):
        if data[i]["latness"] > 0:
            feasible  = False
            break
    
    return data, feasible


def edf(data):
    unsortedTasks = data.values()
    unsortedTasks = sorted(unsortedTasks, key = lambda i: i['dt'])
    sortedTasks = []
    curr_time = 0
    while unsortedTasks:
        n = 0
        while n < len(unsortedTasks):
            bestTask = unsortedTasks[n]
            if bestTask["at"] <= curr_time:
                break
            n +=1
        if bestTask["at"] > curr_time:
            curr_time = bestTask["at"]
        print(bestTask , curr_time)
        bestTask["st"] = curr_time
        curr_time += bestTask['bt']
        bestTask["ct"] = curr_time
        bestTask["latness"] =  bestTask["at"] - curr_time
        sortedTasks.append(bestTask)
        unsortedTasks.pop(unsortedTasks.index(bestTask))
    return sortedTasks
    
def Schedulablity():
    """
    Calculates the utilization factor of the tasks to be scheduled
    and then checks for the schedulablity and then returns true is
    schedulable else false.
    """
    global U_factor
    for i in tasks.keys():
        print(i)
        T.append(int(tasks[i]["Period"]))
        C.append(int(tasks[i]["WCET"]))
        u = int(C[-1])/int(T[-1])
        U.append(u)

    U_factor = sum(U)
    if U_factor<=1:
        print("\nUtilization factor: ",U_factor, "underloaded tasks")

        sched_util = n*(2**(1/n)-1)
        print("Checking condition: ",sched_util)

        count = 0
        T.sort()
        for i in range(len(T)):
            if T[i]%T[0] == 0:
                count = count + 1

        # Checking the schedulablity condition
        if U_factor <= sched_util or count == len(T):
            print("\n\tTasks are schedulable by Rate Monotonic Scheduling!")
            return True
        else:
            print("\n\tTasks are not schedulable by Rate Monotonic Scheduling!")
            return False
    print("\n\tOverloaded tasks!")
    print("\n\tUtilization factor > 1")
    return False

def estimatePriority(RealTime_task):
	"""
	Estimates the priority of tasks at each real time period during scheduling
	"""
	tempPeriod = hp
	P = -1    #Returns -1 for idle tasks
	for i in RealTime_task.keys():
		if (RealTime_task[i]["WCET"] != 0):
			if (tempPeriod > RealTime_task[i]["Period"] or tempPeriod > tasks[i]["Period"]):
				tempPeriod = tasks[i]["Period"] #Checks the priority of each task based on period
				P = i
	return P


def Hyperperiod():
	"""
	Calculates the hyper period of the tasks to be scheduled
	"""
	temp = []
	for i in tasks.keys():
		temp.append(int(tasks[i]["Period"]))
	HP = temp[0]
	for i in temp[1:]:
		HP = HP*i//gcd(HP, i)
	print ("\n Hyperperiod:",HP)
	return HP

def Simulation(hp):
    """
    The real time schedulng based on Rate Monotonic scheduling is simulated here.
    """

    # Real time scheduling are carried out in RealTime_task
    global RealTime_task
    RealTime_task = copy.deepcopy(tasks)
    # validation of schedulablity neessary condition
    for i in RealTime_task.keys():
        RealTime_task[i]["DCT"] = RealTime_task[i]["WCET"]
        if (RealTime_task[i]["WCET"] > RealTime_task[i]["Period"]):
            print(" \n\t The task can not be completed in the specified time ! ", i )

    # main loop for simulator
    print("d",dList)
    try:
        for t in range(hp):

            # Determine the priority of the given tasks
            priority = estimatePriority(RealTime_task)

            if (priority != -1):    #processor is not idle
                print("\nt{}-->t{} :TASK{}".format(t,t+1,priority))
                # Update WCET after each clock cycle
                RealTime_task[priority]["WCET"] -= 1
                # For the calculation of the metrics
                dList["TASK_%d"%int(priority)]["start"].append(t)
                dList["TASK_%d"%int(priority)]["finish"].append(t+1)
                # For plotting the results
                y_axis.append("TASK%d"%int(priority))
                from_x.append(t)
                to_x.append(t+1)

            else:    #processor is idle
                print("\nt{}-->t{} :IDLE".format(t,t+1))
                # For the calculation of the metrics
                dList["TASK_IDLE"]["start"].append(t)
                dList["TASK_IDLE"]["finish"].append(t+1)
                # For plotting the results
                y_axis.append("IDLE")
                from_x.append(t)
                to_x.append(t+1)

            # Update Period after each clock cycle
            for i in RealTime_task.keys():
                RealTime_task[i]["Period"] -= 1
                if (RealTime_task[i]["Period"] == 0):
                    RealTime_task[i] = copy.deepcopy(tasks[i])

            with open('RM_sched.json','w') as outfile2:
                json.dump(dList,outfile2,indent = 4)
    except:
        pass


def drawGantt():
    """
    The scheduled results are displayed in the form of a
    gantt chart for the user to get better understanding
    """
    colors = ['red','green','blue','orange','yellow']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # the data is plotted from_x to to_x along y_axis
    ax = plt.hlines(y_axis, from_x, to_x, linewidth=20, color = colors[n-1])
    plt.title('Rate Monotonic scheduling')
    plt.grid(True)
    plt.xlabel("Real-Time clock")
    plt.ylabel("HIGH------------------Priority--------------------->LOW")
    plt.xticks(np.arange(min(from_x), max(to_x)+1, 1.0))
    plt.show()


def showMetrics():
    """
    Displays the resultant metrics after scheduling such as
    average response time, the average waiting time and the
    time of first deadline miss
    """
    N = []
    startTime = []
    releaseTime = []
    finishTime = []
    avg_respTime = []
    avg_waitTime = []

    # Calculation of number of releases and release time
    for i in tasks.keys():
    
        release =int(hp)/int(tasks[i]["Period"])
        N.append(release)
        temp = []
        for j in range(int(N[int(i)-1])):
            temp.append(j*int(tasks[i]["Period"]))
        # temp.append(hp)
        releaseTime.append(temp)		

    # Calculation of start time of each task
    for j,i in enumerate(tasks.keys()):
        start_array,end_array = filter_out(dList["TASK_%d"%int(i)]["start"],dList["TASK_%d"%int(i)]["finish"],N[j])
        startTime.append(start_array)
        finishTime.append(end_array)

    # Calculation of average waiting time and average response time of tasks
    for i in tasks.keys():
        avg_waitTime.append(st.mean([a_i - b_i for a_i, b_i in zip(startTime[int(i)],releaseTime[int(i)])]))
        avg_respTime.append(st.mean([a_i - b_i for a_i, b_i in zip(finishTime[int(i)],releaseTime[int(i)])]))

    # Printing the resultant metrics
    for i in tasks.keys():
        metrics[i]["Releases"] = N[int(i)]
        metrics[i]["Period"] = tasks[i]["Period"]
        metrics[i]["WCET"] = tasks[i]["WCET"]
        metrics[i]["AvgRespTime"] = avg_respTime[i]
        metrics[i]["AvgWaitTime"] = avg_waitTime[i]
        
        print("\n Number of releases of task %d ="%i,int(N[int(i)]))
        print("\n Release time of task%d = "%i,releaseTime[int(i)])
        print("\n start time of task %d = "%i,startTime[int(i)])
        print("\n finish time of task %d = "%i,finishTime[int(i)])
        print("\n Average Response time of task %d = "%i,avg_respTime[int(i)])
        print("\n Average Waiting time of task %d = "%i,avg_waitTime[int(i)])
        print("\n")

    # Storing results into a JSON file
    with open('Metrics.json','w') as f:
        json.dump(metrics,f,indent = 4)
    print("\n\n\t\tScheduling of %d tasks completed succesfully...."%n)


def filter_out(start_array,finish_array,release_time):
    """A filtering function created to create the required data struture from the simulation results"""
    new_start = []
    new_finish = []
    print(start_array)
    beg_time = min(start_array)
    diff = int(hp/release_time)
    # Calculation of finish time and start time from simulation results
    if(release_time>1):
        new_start.append(beg_time)
        prev = beg_time
        for i in range(int(release_time-1)):
            beg_time = beg_time + diff
            new_start.append(beg_time)
            count = start_array.index(prev)
            for i in range(start_array.index(prev),start_array.index(beg_time)-1):
                    count+=1
            new_finish.append(finish_array[count])
            prev = beg_time
        new_finish.append(max(finish_array))

    else:
        end_time = max(finish_array)
        new_start.append(beg_time)
        new_finish.append(int(end_time))
    return new_start,new_finish



def rm(data):
    global n
    global hp
    global tasks
    global dList
    dList = {}

    tasks = data
    n = len(data)
    for  i in range(n):
        dList["TASK_%d"%i] = {"start":[],"finish":[]}
    dList["TASK_IDLE"] = {"start":[],"finish":[]}
    sched_res = Schedulablity()
    if sched_res == True:
        hp = Hyperperiod()
        Simulation(hp)
        #showMetrics()
    return RealTime_task
def first_come_first_serve(data):
    arrival_time = []
    indx = 0
    for i in data:
        arrival_time.append([i['at'], i['id'], indx])
        indx += 1
    arrival_time.sort()
    curr_time = 0
    for val in arrival_time:
        if data[val[2]]['at'] > curr_time:
            curr_time += data[val[2]]['at'] - curr_time
        curr_time += data[val[2]]['bt']
        data[val[2]]['ct'] = curr_time
    return data


def round_robin(data, tq):
    rr = []
    indx = 0
    curr_time = 0
    flag = True
    for dct in data:
        rr.append([dct['at'], dct['bt'], indx])
        indx += 1
    while flag:
        for i in range(len(rr)):
            bt = rr[i][1]
            if bt != 0 and rr[i][0] < curr_time or curr_time == 0:
                if bt > tq:
                    rr[i][1] -= tq
                    curr_time += tq
                else:
                    rr[i][1] = 0
                    curr_time += bt
                    data[rr[i][2]]['ct'] = curr_time
        flg = True
        for i in rr:
            if not i[1] == 0:
                flg = False
        if flg:
            flag = False
    return data

def edd_helper(processed, task):
    return max(processed+ task["bt"], task["at"])

def edd(data):
    # at -> deadline , bt -> worst computation time
    unsortedTasks = data.copy()
    sortedTasks = []
    processed = 0
    curr_time = 0
    for i in range(len(unsortedTasks)):
        bestTask = unsortedTasks[list(unsortedTasks.keys())[0]]
        bestMdd = edd_helper(processed, bestTask)
        for task in list(unsortedTasks):
            mdd = edd_helper(processed, unsortedTasks[task])
            if mdd < bestMdd:
                bestMdd = mdd
                bestTask = unsortedTasks[task]
        bestTask["st"] = curr_time
        curr_time += bestTask['bt']
        bestTask["ct"] = curr_time
        bestTask["latness"] =  bestTask["at"] - curr_time
        sortedTasks.append(bestTask)
        del unsortedTasks[str(bestTask["id"])]
    return sortedTasks
    

# cyclic executive
def ce(data):
    # at -> deadline , bt -> worst computation time
    unsortedTasks = data.copy()
    sortedTasks = []
    processed = 0
    curr_time = 0
    for i in range(len(unsortedTasks)):
        bestTask = unsortedTasks[list(unsortedTasks.keys())[0]]
        bestTask["st"] = curr_time
        curr_time += bestTask['bt']
        bestTask["ct"] = curr_time
        bestTask["latness"] =  bestTask["at"] - curr_time
        sortedTasks.append(bestTask)
        #del unsortedTasks[str(bestTask["id"])]
    for i in range(len(unsortedTasks)):
        bestTask = unsortedTasks[list(unsortedTasks.keys())[0]]
        bestTask["st"] = curr_time
        curr_time += bestTask['bt']
        bestTask["ct"] = curr_time
        bestTask["latness"] =  bestTask["at"] - curr_time
        sortedTasks.append(bestTask)
    return sortedTasks
    
def shortest_remaining_time(data):
    def find_min_rt_arrived(curr_time, lst):
        lst.sort(key=lambda x: x[1])
        for i in lst:
            if curr_time >= i[0]:
                return i[2], True

    def all_done(lst):
        for i in lst:
            if i[1] != 0:
                return False
        return True

    def reduce_bt(indx, lst):
        for i in range(len(lst)):
            if lst[i][-1] == indx:
                lst[i][1] -= 1
                if lst[i][1] == 0:
                    del lst[i]
                    return lst, True
        return lst, False
    srtf = []
    indx = 0
    curr_time = 0
    for dct in data:
        srtf.append([dct['at'], dct['bt'], indx])
        indx += 1
    srtf.sort()
    while not all_done(srtf):
        index, has_arrived = find_min_rt_arrived(curr_time, srtf)
        if has_arrived:
            curr_time += 1
            srtf, is_done = reduce_bt(index, srtf)
            if is_done:
                data[index]['ct'] = curr_time
    return data


def priority_non_preemptive(data):
    def has_arrived(lst, curr_time, index):
        for i in lst:
            if i[-1] == index:
                if i[1] <= curr_time:
                    return True
        return False
    prior = []
    indx = 0
    curr_time = 0
    for i in data:
        prior.append([i['pr'], i['at'], i['bt'], indx])
        indx += 1
    prior.sort(reverse=True)
    while prior:
        for val in prior:
            if has_arrived(prior, curr_time, val[-1]):
                curr_time += val[2]
                data[val[-1]]['ct'] = curr_time
                prior.remove(val)
    return data


def priority_preemptive(data):
    def find_max_prior_arrived(curr_time, lst):
        lst.sort(reverse=True)
        for i in lst:
            if curr_time >= i[1]:
                return i[-1], True

    def all_done(lst):
        for i in lst:
            if i[2] != 0:
                return False
        return True

    def reduce_bt(indx, lst):
        for i in range(len(lst)):
            if lst[i][-1] == indx:
                lst[i][2] -= 1
                if lst[i][2] == 0:
                    del lst[i]
                    return lst, True
        return lst, False
    prior = []
    indx = 0
    curr_time = 0
    for i in data:
        prior.append([i['pr'], i['at'], i['bt'], indx])
        indx += 1
    prior.sort(reverse=True)
    while not all_done(prior):
        index, has_arrived = find_max_prior_arrived(curr_time, prior)
        if has_arrived:
            curr_time += 1
            prior, is_done = reduce_bt(index, prior)
            if is_done:
                data[index]['ct'] = curr_time
    return data
