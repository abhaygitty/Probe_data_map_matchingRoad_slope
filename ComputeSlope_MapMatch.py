import numpy as np
import pickle
import os
from multiprocessing import Process
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
# import time
from datetime import datetime, timezone, timedelta
import json

def datetimeloading(dir): # loadTime
    x= None
    y= None
    if not os.path.exists('probeID.pckl') or not os.path.exists('probeTime.pckl'):
        x,y = np.loadtxt(dir+'/Partition6467ProbePoints.csv', dtype=str, delimiter=',', usecols=(0,1), unpack=True)
        x = np.array([fixString(x[i]) for i in range(x.shape[0])])
        y = np.array([datetime.strptime(y[i], "b'%m/%d/20%y %I:%M:%S %p'").timestamp() for i in range(y.shape[0])])
        pickle.dump(x,open('probeID.pckl','wb'))
        pickle.dump(y,open('probeTime.pckl','wb'))
    else:
        x = pickle.load(open('probeID.pckl','rb'))
        y = pickle.load(open('probeTime.pckl','rb'))
    return x,y

def speeddata(dir):
    if os.path.exists('probeSpeed.pckl'):
        speed = pickle.load(open('probeSpeed.pckl','rb'))
        return speed
    speed = np.loadtxt(dir+'/Partition6467ProbePoints.csv', delimiter=',', usecols=(6,))
    speed = 0.621371 * speed
    pickle.dump(speed, open('probeSpeed.pckl','wb'))
    return speed

def probeMovementPointer(dir): #to get determine the angle of movement of the probe vehicle with respect to the road segment or link
    if os.path.exists('probeHeading.pckl'):
        heading = pickle.load(open('probeHeading.pckl','rb'))
        return heading
    heading = np.loadtxt(dir+'/Partition6467ProbePoints.csv', delimiter=',', usecols=(7,))
    pickle.dump(heading, open('probeHeading.pckl','wb'))
    return heading

def loadProbeLatLong(dir):
    x=y= None
    if not os.path.exists('probeX.pckl') or not os.path.exists('probeY.pckl'):
        x,y = np.loadtxt(dir+'/Partition6467ProbePoints.csv', delimiter=',', usecols=(3,4), unpack=True)
        pickle.dump(x,open('probeX.pckl','wb'))
        pickle.dump(y,open('probeY.pckl','wb'))
    else:
        x = pickle.load(open('probeX.pckl','rb'))
        y = pickle.load(open('probeY.pckl','rb'))
    return x,y

def loadLinkDOT(dir):
    if os.path.exists('linkDOT.json'):
        dot = json.load(open('linkDOT.json','r'))
        return dot
    ids, dot = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,5), unpack=True)
    dot = [fixString(dot[i]) for i in range(dot.shape[0])]
    ids = [fixString(ids[i]) for i in range(ids.shape[0])]
    D = dict([(i,d) for i,d in zip(ids,dot)])
    json.dump(D,open('linkDOT.json','w'))
    return D

def createP1P2(ids,l_x,l_y,dot):
    if os.path.exists('linkP1.pckl') and os.path.exists('linkP2.pckl') and os.path.exists('linkP1P2ID.pckl'):
        P1 = pickle.load(open('linkP1.pckl','rb'))
        P2 = pickle.load(open('linkP2.pckl','rb'))
        lid = pickle.load(open('linkP1P2ID.pckl','rb'))
        return lid, P1, P2

    P1 = []
    l_id = []
    uni, index, count = np.unique(ids, return_counts=True, return_index=True)
    for i in range(uni.shape[0]):
        ID = uni[i]
        x = l_x[index[i]:index[i]+count[i]]
        y = l_y[index[i]:index[i]+count[i]]
        if dot[ID] == 'F':
            [(P1.append([x[k], y[k]]), l_id.append(ID)) for k in range(x.shape[0])]
        elif dot[ID] == 'T':
            [(P1.append([x[k], y[k]]), l_id.append(ID)) for k in range(x.shape[0]-1,-1,-1)]
        elif dot[ID] == 'B':
            [(P1.append([x[k], y[k]]), l_id.append(ID)) for k in range(x.shape[0]-1,-1,-1)]
            [(P1.append([x[k], y[k]]), l_id.append(ID)) for k in range(x.shape[0])]
    l_id = np.asarray(l_id)
    P1 = np.asarray(P1)
    P2 = P1[1:]
    l2 = l_id[1:]
    P1 = P1[:-1]
    l1 = l_id[:-1]

    ind = np.where((l1 == l2) & ~((P1[:,0] == P2[:,0]) & (P1[:,1] == P2[:,1])))
    P1 = P1[ind]
    P2 = P2[ind]
    l1 = l1[ind]
    pickle.dump(P1,open('linkP1.pckl','wb'))
    pickle.dump(P2,open('linkP2.pckl','wb'))
    pickle.dump(l1,open('linkP1P2ID.pckl','wb'))
    return l1, P1, P2

def loadLinkLatLong(dir):
    if os.path.exists('linkX.pckl') and os.path.exists('linkY.pckl') and os.path.exists('linkID.pckl'):
        ID = pickle.load(open('linkID.pckl','rb'))
        X = pickle.load(open('linkX.pckl','rb'))
        Y = pickle.load(open('linkY.pckl','rb'))
        return ID, X, Y
    ID,x = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,14,), unpack=True)
    x = np.array([fixString(x[i]) for i in range(x.shape[0])])
    ID = np.array([fixString(ID[i]) for i in range(ID.shape[0])])
    IDx = defaultdict(lambda: [])
    IDy = defaultdict(lambda: [])
    for i in range(x.shape[0]):
        temp = x[i].split('|')
        for comb in temp:
            comb = comb.split('/')
            lat = float(comb[0])
            lng = float(comb[1])
            IDx[ID[i]].append(lat)
            IDy[ID[i]].append(lng)
    del ID
    ID, X, Y = getLinkXYArray(IDx, IDy)
    pickle.dump(ID, open('linkID.pckl','wb'))
    pickle.dump(X, open('linkX.pckl','wb'))
    pickle.dump(Y, open('linkY.pckl','wb'))
    return ID, X, Y

def loadLinkSlope(dir):
    if os.path.exists('linkSlopes.json'):
        slope = json.load(open('linkSlopes.json','r'))
        return slope
    ids, slopeinfo = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,16), unpack=True)
    ids = [fixString(ids[i]) for i in range(ids.shape[0])]
    slopeinfo = [fixString(s) for s in slopeinfo]
    print(len(ids),len(slopeinfo))
    slopes = defaultdict(lambda: [])
    for s,i in zip(slopeinfo,ids):
        if s != '':
            y = s.split('|')
            for x in y:
                t = float(x.split('/')[1])
                slopes[i].append(t)
    json.dump(slopes, open('linkSlopes.json','w'))
    return slopes

def getLinkXYArray(X, Y):
    IDs = []
    Xs = []
    Ys = []
    for k in X:
        for x,y in zip(X[k],Y[k]):
            IDs.append(k)
            Xs.append(x)
            Ys.append(y)
    IDs = np.asarray(IDs)
    Xs = np.asarray(Xs, dtype=np.float64)
    Ys = np.asarray(Ys, dtype=np.float64)
    return IDs, Xs, Ys

def fixString(x):
    res = ''
    try:
        res = x.split("'")[-2]
    except Exception as e:
        print('Exception ',x, x.split("'"))
        raise e
    finally:
        return res

def loadLinkIdentifiers(dir):
    ids, ref, nref = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,1,2), unpack=True)
    idref = defaultdict(lambda: [])
    for i in range(ref.shape[0]):
        x = fixString(ids[i])
        y = fixString(ref[i])
        z = fixString(nref[i])
        idref[x].append((y,z))
    return idref

def loadLinkLength(dir):
    if os.path.exists('lID.pckl') and os.path.exists('linkLengths.pckl'):
        x = pickle.load(open('lID.pckl','rb'))
        dist = pickle.load(open('linkLengths.pckl','rb'))
        return x, dist
    x = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,), unpack=True)
    dist = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=float, delimiter=',', usecols=(3,))
    x = np.array([fixString(x[i]) for i in range(x.shape[0])])
    pickle.dump(x,open('lID.pckl','wb'))
    pickle.dump(dist,open('linkLengths.pckl','wb'))
    return x, dist	

def loadLink(dir):
    x,y,z,cat = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,1,2,5), unpack=True)
    dist = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=float, delimiter=',', usecols=(3,))
    graph = defaultdict(lambda: [])
    lengths = defaultdict(lambda: [])
    for i in range(y.shape[0]):
        Y = fixString(y[i])
        Z = fixString(z[i])
        C = fixString(cat[i])
        ID = fixString(x[i])
        if C == 'F':
            graph[Y].append(Z)
            lengths[Y].append((Z,dist[i],ID))
        elif C == 'T':
            graph[Z].append(Y)
            lengths[Z].append((Y,dist[i],ID))
        elif C == 'B':
            graph[Y].append(Z)
            lengths[Y].append((Z,dist[i],ID))
            graph[Z].append(Y)
            lengths[Z].append((Y,dist[i],ID))

    return graph, lengths

def timingInfo(ids, date_time):
    slots = {}
    if not os.path.exists('slots.json'):
        sind = date_time.argsort()
        ids1 = ids[sind]
        dt = date_time[sind]
        slots = defaultdict(lambda: defaultdict(lambda: []))
        i = 0
        proc=0.
        while i < dt.shape[0]:
            time = dt[i]
            r = np.where((dt >= time) & (dt < (time+600)))
            r = r[0]
            if r.shape[0] > 1:
                uni_ids, counts = np.unique(ids1[r], return_counts=True)
                x = np.where(counts > 1)[0]
                uni_ids = uni_ids[x].tolist()
                for j in range(r.shape[0]):
                    if ids1[r[j]] in uni_ids:
                        slots[time][ids1[r[j]]].append(dt[r[j]])
            if r.shape[0] > 0:
                i += r.shape[0]
            else:
                i += 1
            proc = (float(i)/dt.shape[0])*100
        slots = OrderedDict(dict(slots))
        json.dump(slots,open('slots.json','w'))
    else:
        print('Loading "slots.json" ...')
        slots = json.load(open('slots.json','r'), object_pairs_hook=OrderedDict)
    return slots

def loadData(dir):
    pTime = []
    pLatLong = []
    pAlt = []
    pVelocity = []
    lData = []
    if not os.path.exists('pTime.pckl'):
        pTime = np.loadtxt(dir+'/Partition6467ProbePoints.csv',dtype=str,delimiter=',',usecols=(0,1))
        pLatLong = np.loadtxt(dir+'/Partition6467ProbePoints.csv',dtype=str,delimiter=',',usecols=(0,3,4))
        pAlt = np.loadtxt(dir+'/Partition6467ProbePoints.csv',dtype=str,delimiter=',',usecols=(0,5))
        pVelocity = np.loadtxt(dir+'/Partition6467ProbePoints.csv',dtype=str,delimiter=',',usecols=(0,6,7))
        pickle.dump(pTime,open('pTime.pckl','wb'))
        pickle.dump(pLatLong,open('pLatLong.pckl','wb'))
        pickle.dump(pAlt,open('pAlt.pckl','wb'))
        pickle.dump(pVelocity,open('pVelocity.pckl','wb'))
    else:
        pTime = pickle.load(open('pTime.pckl','rb'))
        pLatLong = pickle.load(open('pLatLong.pckl','rb'))
        pAlt = pickle.load(open('pAlt.pckl','rb'))
        pVelocity = pickle.load(open('pVelocity.pckl','rb'))
    return pTime, pLatLong, pAlt, pVelocity
    
    def findAngle(P1, P2):
        arctan = lambda x,y: np.arctan2(x,y)
        pi = np.pi
        delta = P2-P1
        alpha = np.zeros((P1.shape[0],),dtype=P1.dtype)
        ind_t = np.where((delta[:,0] < 0) & (delta[:,1] >= 0))[0]
        ind_f = np.where(~((delta[:,0] < 0) & (delta[:,1] >= 0)))[0]
        alpha[ind_t] = (2.5*pi - arctan(delta[ind_t,1],delta[ind_t,0])) * 180./pi
        alpha[ind_f] = (0.5*pi - arctan(delta[ind_f,1],delta[ind_f,0])) * 180./pi
        return alpha

def findPerpDist(P1, P2, P3):
    x3, y3 = P3
    ER = 3956.
    p3 = np.array([x3,y3])
    p1_p2 = np.sum(np.square(P2-P1),axis=1)
    x = np.sum((p3-P1)*(P2-P1),axis=1)
    ind = np.where(p1_p2 != 0)[0]
    _mu = np.zeros(p1_p2.shape,dtype=p1_p2.dtype)
    _mu[ind] = x[ind]/p1_p2[ind]
    p = P1 + np.vstack((_mu,_mu)).T*(P2-P1)
    pi = np.pi
    
    R = p*(pi/180.)
    R3 = p3*(pi/180.)
    ab = R3-R
    arcsin = lambda x: np.arcsin(x)
    sin = lambda x: np.sin(x)
    cos = lambda x: np.cos(x)
    PD = ER * arcsin(np.sqrt( sin(ab[:,0]/2.)**2 + cos(R3[0])*cos(R[:,0])*(sin(ab[:,1]/2.))**2) )
    return PD

def findDirection(alpha, heading):
    HE = np.absolute(heading - alpha)
    ind_great = np.where(HE > 180)[0]
    HE[ind_great] = 360. - HE[ind_great]
    return HE

def createCandidate(P, l_id, P1, P2, p_speed, p_head, alpha):
    PDs = findPerpDist(P1, P2, P)
    HEs = findDirection(alpha, p_head)
    sind = PDs.argsort()
    PDs = PDs[sind]
    HEs = HEs[sind]
    IDs = l_id[sind]
    p1 = P1[sind,:]

    if p_speed < 7.:
        if IDs.shape[0] >= 4:
            return [(IDs[i], p1[i,0], p1[i,1]) for i in range(4)]
        else:
            return [(IDs[i], p1[i,0], p1[i,1]) for i in range(IDs.shape[0])]
    else:
        ind = np.where(HEs <= 90)[0]
        if ind.shape[0] > 0:
            PDs = PDs[ind]
            IDs = IDs[ind]
            p1 = p1[ind,:]
            if ind.shape[0] >= 4:
                return [(IDs[i], p1[i,0], p1[i,1]) for i in range(4)]
            else:
                return [(IDs[i], p1[i,0], p1[i,1]) for i in range(ind.shape[0])]
        else:
            return []

def Pairsgen(p1, p2):
    if len(p1) == 0 or len(p2) == 0:
        return []
    else:
        pairs = []
        for i in p1:
            for j in p2:
                pairs.append((i,j))
        return pairs

def subNode(node1, node2, graph, visited, result, level):
    if not visited[node1]:
        visited[node1] = True
        children = graph[node1]
        if level > 4:
            result[node1] = False
            return False
        if len(children) == 0:
            result[node1] = False
            return False
        if node2 in children:
            result[node1] = True
            return True
        else:
            for i in children:
                if subNode(i,node2,graph,visited,result, level+1):
                    result[node1] = True
                    return True
            result[node1] = False
            return False
    else:
        return result[node1]

def linkExist(id1, id2, graph, lids, dot):
    if id1 == id2:
        return True
    else:
        ref1, nref1 = lids[id1][0]
        ref2, nref2 = lids[id2][0]
        visited = defaultdict(lambda: False)
        result = defaultdict(lambda: False)
        if dot[id1] == 'F' and dot[id2] =='T':
            return subNode(nref1, nref2, graph, visited, result, 0)
        elif dot[id1] == 'T' and dot[id2] =='F':
            return subNode(ref1, ref2, graph, visited, result, 0)			
        elif dot[id1] == 'F' and dot[id2] == 'F':
            return subNode(nref1, ref2, graph, visited, result, 0)
        elif dot[id1] == 'T' and dot[id2] == 'T':
            return subNode(ref1, nref2, graph, visited, result, 0)
        elif dot[id1] == 'B' and dot[id2] != 'B':
            if dot[id2] == 'T':
                a = subNode(ref1, nref2, graph, visited, result, 0)
                b = subNode(nref1, nref2, graph, visited, result, 0)
                return a or b
            elif dot[id2] == 'F':
                a = subNode(ref1, ref2, graph, visited, result, 0)
                b = subNode(nref1, ref2, graph, visited, result, 0)
                return a or b
        elif dot[id2] == 'B' and dot[id1] != 'B':
            if dot[id1] == 'T':
                a = subNode(ref1, ref2, graph, visited, result, 0)
                b = subNode(ref1, nref2, graph, visited, result, 0)
                return a or b
            elif dot[id1] == 'F':
                a = subNode(nref1, ref2, graph, visited, result, 0)
                b = subNode(nref1, nref2, graph, visited, result, 0)
                return a or b
        elif dot[id2] == 'B' and dot[id1] == 'B':
            a = subNode(ref1, nref2, graph, visited, result, 0)
            b = subNode(nref1, ref2, graph, visited, result, 0)
            c = subNode(nref1, nref2, graph, visited, result, 0)
            d = subNode(ref1, ref2, graph, visited, result, 0)
            return a or b or c or d

def temNode(slot_data, l_id, P1, P2, p_speed, p_head, alpha):
    p_x, p_y = slot_data
    Lat = 0.001
    dLong = 0.001
    if p_x.shape[0] <2:
        return {}
    CandidateLinksidates = defaultdict(lambda: [])
    for i in range(p_x.shape[0]):
        x = p_x[i]
        y = p_y[i]

        rxMin = x - dLat
        rxMax = x + dLat
        ryMin = y - dLong
        ryMax = y + dLong
        ind = np.where(((P1[:,0] >= rxMin) & (P1[:,0] <= rxMax)) & ((P1[:,1] >= ryMin) & (P1[:,1] <= ryMax)))[0]
        if ind.shape[0] == 0:
            candidates[str(x)+','+str(y)] = []
            continue
        p1_f = P1[ind,:]
        p2_f = P2[ind,:]
        lid_f = l_id[ind]
        alpha_f = alpha[ind]
        candidates[str(x)+','+str(y)] = createCandidate((x,y), lid_f, p1_f, p2_f, p_speed[i], p_head[i], alpha_f)
    return candidates

def genRoute(slot, slot_ID, graph, dot, lidref, p_id, times, d_t, p_x, p_y):
    if os.path.exists('CandidateLinks/{}.json'.format(slot_ID)):
        return json.load(open('CandidateLinks/{}.json'.format(slot_ID),'r'))
    candidateSet = {}
    for i,car in enumerate(slot):
        time = sorted(times[car])
        ind = np.where((p_id == car) & ((d_t >= time[0]) & (d_t <= time[-1])))[0]
        x = p_x[ind]
        y = p_y[ind]
        coors = ['{},{}'.format(str(x[k]),str(y[k])) for k in range(x.shape[0])]
        c = []
        [c.append(k) for k in coors if k not in c]
        del coors[:]
        coors = c
        pvid = []
        indices = []
        for index,c in enumerate(coors):
            pvid.append([ID[0] for ID in slot[car][c]])
            if len(slot[car][c]) > 0:
                indices.append(index)
        pvid = [pvid[ix] for ix in indices]
        coors = [coors[ix] for ix in indices]
        pvid1 = pvid[:-1]
        pvid2 = pvid[1:]
        routes = []
        for p1, p2 in zip(pvid1, pvid2):
            pairs = Pairsgen(p1,p2)
            before = len(pairs)
            for a,b in pairs:
                if not linkExist(a,b,graph,lidref,dot):
                    pairs.remove((a,b))
            routes.append(pairs)
        candidateSet[car] = (routes, coors)
    return candidateSet

def MapMatching(p_id, d_t, p_x, p_y, slots, l_id, P1, P2, p_speed, p_head, alpha, Pname = 'Main'):
    x = None
    prog = 0.
    tot = len(slots)
    
    for j,k in enumerate(slots):
        if os.path.exists('slot_cand/{}.json'.format(k)):
            
            prog = (j/(float(tot)-1.)) * 100
            print('\rCompleted : {:.2f}%, Process: {}'.format(prog, Pname),end=' ')
            continue
        cand = {}
        for y,i in enumerate(slots[k]):
            ind = np.where((p_id == i) & ((d_t >= slots[k][i][0]) & (d_t <= slots[k][i][-1])))[0]
            x = temNode((p_x[ind], p_y[ind]), l_id, P1, P2, p_speed[ind], p_head[ind], alpha)
            cand[i] = x
        prog = (j/(float(tot)-1.)) * 100
        print('\rCompleted : {:.2f}%, Process: {}'.format(prog, Pname),end=' ')
        del cand

def candidateGen(p_id, d_t, p_x, p_y, slots, dot, graph, lidref):
    prog = 0.
    tot = len(slots)
    
    for j,k in enumerate(slots):
        if not os.path.exists('CandidateLinks/{}.json'.format(k)):
            
            slot = json.load(open('slot_cand/{}.json'.format(k),'r'))
            slot_ID = k
            times = slots[slot_ID]
            links = genRoute(slot, slot_ID, graph, dot, lidref, p_id, times, d_t, p_x, p_y)
            json.dump(links, open('CandidateLinks/{}.json'.format(k),'w'))
            del links
            prog = (j/(float(tot)-1.)) * 100
            print('\rCompleted : {:.2f}%'.format(prog),end=' ')
        else:
            prog = (j/(float(tot)-1.)) * 100
            print('\rCompleted : {:.2f}%'.format(prog),end=' ')
            
if __name__ == '__main__':
    a, b, c = loadLinkLatLong('probe_data_map_matching')
    if os.path.exists('linkX.pckl') and os.path.exists('linkY.pckl') and os.path.exists('linkID.pckl'):
        a = pickle.load(open('linkID.pckl','rb'))
        b = pickle.load(open('linkX.pckl','rb'))
        c = pickle.load(open('linkY.pckl','rb'))
    ID,x = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,14,), unpack=True)
    x = np.array([fixString(x[i]) for i in range(x.shape[0])])
    ID = np.array([fixString(ID[i]) for i in range(ID.shape[0])])
    IDx = defaultdict(lambda: [])
    IDy = defaultdict(lambda: [])
    for i in range(x.shape[0]):
        temp = x[i].split('|')
        for comb in temp:
            comb = comb.split('/')
            lat = float(comb[0])
            lng = float(comb[1])
            IDx[ID[i]].append(lat)
            IDy[ID[i]].append(lng)
    del ID
    a, b, c = getLinkXYArray(IDx, IDy)
    pickle.dump(a, open('linkID.pckl','wb'))
    pickle.dump(b, open('linkX.pckl','wb'))
    pickle.dump(c, open('linkY.pckl','wb'))
    
    
    #sl = loadLinkSlope('probe_data_map_matching')
    
    if os.path.exists('linkSlopes.json'):
        slope = json.load(open('linkSlopes.json','r'))
        return slope
    ids, slopeinfo = np.loadtxt(dir+'/Partition6467LinkData.csv', dtype=str, delimiter=',', usecols=(0,16), unpack=True)
    ids = [fixString(ids[i]) for i in range(ids.shape[0])]
    slopeinfo = [fixString(s) for s in slopeinfo]
    sl = defaultdict(lambda: [])
    for s,i in zip(slopeinfo,ids):
        if s != '':
            y = s.split('|')
            for x in y:
                t = float(x.split('/')[1])
                sl[i].append(t)
    json.dump(sl, open('linkSlopes.json','w'))

    for i,k in enumerate(sl):
        if i == 2:
            break
        ind = np.where(a == k)[0]
        print(len(sl[k]) == ind.shape[0])
        print(sl[k])
    
    #calling loaddatetime
    dat = 'probe_data_map_matching'
    p_id, d_t = datetimeloading(dat)
    
    
    slots = timingInfo(p_id, d_t)
    
    p_x, p_y = loadProbeLatLong(dat)
    
    
    l_id, l_x, l_y = loadLinkLatLong(dat)
    
    p_speed = speeddata(dat)
    
    p_head = probeMovementPointer(dat)
    
    l_id, P1, P2 = createP1P2(l_id, l_x, l_y, dot)
    
    lidref = loadLinkIdentifiers(dat)
    
    graph = loadLink(dat)[0]

    
    
    alpha = findAngle(P1,P2)
    
    x = len(slots)    
    
    part = int(x/4)
    
    slots = OrderedDict(sorted(list(slots.items()), key=lambda x: x[0]))
    

    x = len(slots)
    part = int(x/4)
    t1 = Process(target = MapMatching, args=(p_id, d_t, p_x, p_y, OrderedDict(list(slots.items())[:part]), l_id, P1, P2, p_speed, p_head, alpha, 'P1'))
    t2 = Process(target = MapMatching, args=(p_id, d_t, p_x, p_y, OrderedDict(list(slots.items())[part : 2*part]), l_id, P1, P2, p_speed, p_head, alpha, 'P2'))
    t3 = Process(target = MapMatching, args=(p_id, d_t, p_x, p_y, OrderedDict(list(slots.items())[2*part : 3*part]), l_id, P1, P2, p_speed, p_head, alpha, 'P3'))
    t4 = Process(target = MapMatching, args=(p_id, d_t, p_x, p_y, OrderedDict(list(slots.items())[3*part:]), l_id, P1, P2, p_speed, p_head, alpha, 'P3'))
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

    candidateGen(p_id, d_t, p_x, p_y, slots, dot, graph, lidref)
