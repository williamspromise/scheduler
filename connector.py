from algorithms import *
import json
algos = {
    '1': edd,
    '2': edf,
    '3': ce,
    '4': rm,
}


def main():
    with open('data.json', "r") as read_file:
        data = json.load(read_file)
    f = open('fl.txt')
    txt = next(f)
    txt = txt.strip('\n')
    if txt == '6':
        tq = next(f)
    f.close()
    if txt == '6':
        res = algos[txt](data, int(tq))
    else:
        print(txt,algos[txt])
        res = algos[txt](data)
    avg,feasible = analyze(res)
    data_f = open('data.json', 'w')
    data_f.write(json.dumps(res))
    data_f.close()
    return avg, feasible
