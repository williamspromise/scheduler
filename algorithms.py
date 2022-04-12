

def analyze(data):
    feasible = True
    for i  in range(len(data)):
        if data[i]["latness"] > 0:
            feasible  = False
            break
    
    return data, feasible


def edf(data):
    unsortedTasks = data.values()
    print(unsortedTasks)
    unsortedTasks = sorted(unsortedTasks, key = lambda i: i['dt'])
    print(unsortedTasks)
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
    

def ce(data):
    pass   

def rm(data):
    pass

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
