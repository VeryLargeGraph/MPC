# -*- coding: utf-8 -*-
import datetime


class Graph:

    def __init__(self, path):
        self.tGraph = self.readGraph(path)  # dic with u:{v1:[time1,time2,...]; v2:...} time sorted!
        self.adjset = {}  # dic with u: set(adj(u))
        self.node_time = {}  # dic with u:[2001,2002,...]  sorted!!!
        self.adj_time_nodes = {}  # dic with 2001:{u1:set(v1,v2,v3), u2:()...}
        self.degree = {}  # dic to check whether one node is existed
        self.PT = {}  # dic with {u1:{(time1,1):[degree1, degree2, degree3], (time2,1):[degree1, degree2, degree3]...}, u2:...}
        self.IPT = {}  # dic with {u1:{time1:[(time1,1),(time2,1)]...}, u2:...}

        print("number of nodes:" + str(len(self.tGraph)))
        number1 = 0
        number2 = 0
        for u in self.tGraph:
            number1 += len(self.tGraph[u])
            for v in self.tGraph[u]:
                number2 += len(self.tGraph[u][v])
        print("number of edges:" + str(number1 / 2))
        print("number of temporal edges:" + str(number2 / 2))

    def prepare(self, theta, k):
        starttime = datetime.datetime.now()
        nodes = self.kcore(k)
        endtime = datetime.datetime.now()
        interval = (endtime - starttime)
        print("kCore time:" + str(interval))
        print("kCore #nodes:" + str(len(nodes)))

        for node in nodes:
            self.adjset[node] = set(self.tGraph[node].keys()) & nodes

            self.node_time[node] = []
            for to_node in self.tGraph[node]:
                if self.degree[to_node] >= k:  # inside nodes
                    self.node_time[node] += self.tGraph[node][to_node]
            self.node_time[node] = sorted(list(set(self.node_time[node])))

            for t in self.node_time[node]:
                list_temp = set()
                for to_node in self.tGraph[node]:
                    if t in self.tGraph[node][to_node]:
                        list_temp.add(to_node)
                if len(list_temp) > 0:
                    if t in self.adj_time_nodes:
                        self.adj_time_nodes[t][node] = set(list_temp)
                    else:
                        self.adj_time_nodes[t] = {}
                        self.adj_time_nodes[t][node] = set(list_temp)

    def reset(self):
        self.adjset = {}
        self.node_time = {}
        self.adj_time_nodes = {}
        self.degree = {}
        self.PT = {}
        self.IPT = {}

    def readGraph(self, path):
        tGraph = {}  # graph like u:{v1:[2001,2003]; v2:...}
        file = open(path)
        while 1:
            lines = file.readlines(500000)
            if not lines:
                break
            for line in lines:
                if line != '\n':
                    line = line.strip('\n').split('\t')
                    from_id, to_id, time_id = int(line[0]), int(line[1]), int(line[2])
                    if from_id == to_id:
                        continue
                    if from_id > to_id:
                        from_id, to_id = to_id, from_id
                    if from_id in tGraph:
                        if to_id not in tGraph[from_id]:
                            tGraph[from_id][to_id] = [time_id]
                        else:
                            tGraph[from_id][to_id].append(time_id)
                    else:
                        tGraph[from_id] = {}
                        tGraph[from_id][to_id] = [time_id]
        file.close()
        print("loading...")

        result = {}
        for f_id in tGraph:
            for t_id in tGraph[f_id]:
                if f_id in result:
                    result[f_id][t_id] = sorted(list(set(tGraph[f_id][t_id])))
                else:
                    result[f_id] = {}
                    result[f_id][t_id] = sorted(list(set(tGraph[f_id][t_id])))
                if t_id in result:
                    result[t_id][f_id] = result[f_id][t_id]
                else:
                    result[t_id] = {}
                    result[t_id][f_id] = result[f_id][t_id]
        return result

    def kcore(self, k):
        q = []
        degree = {}
        D = []
        for u in self.tGraph:
            degree[u] = len(self.tGraph[u])
            if degree[u] < k:
                q.append(u)

        while (len(q) > 0):
            v = q.pop()
            D.append(v)
            for w in self.tGraph[v]:
                if degree[w] >= k:
                    degree[w] = degree[w] - 1
                    if degree[w] < k:
                        q.append(w)
        allowed_nodes = set(self.tGraph.keys()) - set(D)
        self.degree = degree
        return allowed_nodes

    def ComputePeriod(self, theta, k, u, F):
        if len(self.node_time[u]) < theta:
            return False
        max_time_temp = max(self.node_time[u])

        PeriodicQ = []
        StartS = []
        Vc = self.set_union(self.adjset[u], F)

        for t in self.node_time[u]:
            if u in self.adj_time_nodes[t]:
                neighour = self.adj_time_nodes[t][u]
            else:
                continue
            degree = len(self.set_union(neighour, Vc))

            if degree >= k:
                tempQ = PeriodicQ[:]
                for S in tempQ:  # S:[s,i,l]
                    if ((t - S[0]) % S[1]) == 0:
                        if (t - S[0]) / S[1] != S[2]:
                            PeriodicQ.remove(S)
                            continue
                        S[2] = S[2] + 1
                        if S[2] == theta:
                            return True
                    if (t - S[0]) > (theta - 1) * S[1]:
                        PeriodicQ.remove(S)
                for s in StartS:
                    if (t - s) * (theta - 2) <= max_time_temp - t:
                        PeriodicQ.append([s, t - s, 2])
                StartS.append(t)
        return False

    def set_union(self, a, b):
        return a & b

    def ComputePeriodAll(self, theta, k, u, F):
        if len(self.node_time[u]) < theta:
            return {}
        max_time_temp = max(self.node_time[u])

        PeriodicQ = []
        StartS = []
        Vc = self.set_union(self.adjset[u], F)

        PeriodicSet = {}
        for t in self.node_time[u]:
            if u in self.adj_time_nodes[t]:
                neighour = self.adj_time_nodes[t][u]
            else:
                continue
            degree = len(self.set_union(neighour, Vc))
            if degree >= k:
                tempQ = PeriodicQ[:]
                for S in tempQ:
                    if ((t - S[0]) % S[1]) == 0:
                        if (t - S[0]) / S[1] != len(S[2]):
                            PeriodicQ.remove(S)
                            continue
                        S[2].append(degree)
                        if len(S[2]) == theta:
                            PeriodicSet[(S[0], S[1])] = S[2]
                            PeriodicQ.remove(S)
                            continue
                    if (t - S[0]) > (theta - 1) * S[1]:
                        PeriodicQ.remove(S)
                for s in StartS:
                    if (t - s[0]) * (theta - 2) <= max_time_temp - t:
                        PeriodicQ.append([s[0], t - s[0], [s[1], degree]])
                StartS.append([t, degree])
        return PeriodicSet

    def WPCore(self, theta, k):
        Vc = self.kcore(k)
        q = []
        degree = self.degree
        for u in Vc:
            if self.ComputePeriod(theta, k, u, Vc) is False:
                degree[u] = -1
                q.append(u)

        Vc = Vc - set(q)
        while len(q) > 0:
            v = q.pop()
            for w in self.adjset[v]:
                if degree[w] >= k:
                    degree[w] -= 1
                    if degree[w] < k:
                        q.append(w)
                        Vc.remove(w)
                    else:
                        if self.ComputePeriod(theta, k, w, Vc) is False:
                            degree[w] = -1
                            q.append(w)
                            Vc.remove(w)
        return Vc

    def WPCorePlus(self, theta, k):
        Vc = self.kcore(k)
        degree = self.degree

        q = []
        D = set()
        nodes = Vc.copy()
        # degree_n = sorted(degree.items(), key=lambda item: item[1], reverse=False)

        for u in nodes:
            if u in D:
                continue
            self.PT[u] = self.ComputePeriodAll(theta, k, u, Vc)
            if len(self.PT[u]) is 0 or degree[u] < k:
                q.append(u)
                degree[u] = -1

            self.InvertIndex(self.PT[u], u)

            while len(q) > 0:
                v = q.pop()
                D.add(v)
                Vc.remove(v)
                if v in self.PT:
                    del (self.PT[v])
                    del (self.IPT[v])

                for w in self.adjset[v]:
                    if degree[w] >= k:
                        degree[w] -= 1
                        if degree[w] < k:
                            q.append(w)
                            continue
                        if w in self.PT:
                            self.UpdatePeriod(w, v, k)
                            if len(self.PT[w]) is 0:
                                q.append(w)
                                degree[w] = -1
        return Vc - D

    def InvertIndex(self, pt, u):
        self.IPT[u] = {}
        for item in pt:
            for i in range(len(pt[item])):
                i1, i2 = item
                current_t = i1 + i * i2
                if current_t in self.IPT[u]:
                    self.IPT[u][current_t].append(item)
                else:
                    self.IPT[u][current_t] = [item]

    def UpdatePeriod(self, w, v, k):
        for t in self.tGraph[w][v]:
            if t in self.IPT[w]:
                for item in self.IPT[w][t]:
                    if item in self.PT[w]:
                        key = int((t - item[0]) / item[1])
                        self.PT[w][item][key] -= 1
                        if self.PT[w][item][key] < k:
                            del (self.PT[w][item])

    def SPCore(self, theta, k):
        Vc = self.WPCorePlus(theta, k)
        degree = self.degree

        EQ = set()
        ED = set()
        nodes = Vc.copy()
        self.PTcapPTcapEPT = {}

        for u in nodes:
            for v in self.adjset[u]:
                if degree[v] < k:
                    continue
                if u > v:
                    temp1, temp2 = v, u
                else:
                    temp1, temp2 = u, v
                if (temp1, temp2) in ED or (temp1, temp2) in EQ:
                    continue

                n1, n2 = u, v
                if n1 not in self.PTcapPTcapEPT:
                    self.PTcapPTcapEPT[n1] = {}
                    self.PTcapPTcapEPT[n1][n2] = self.PT_cap_PT_cap_EPT(n1, n2, theta)
                else:
                    if n2 not in self.PTcapPTcapEPT[n1]:
                        self.PTcapPTcapEPT[n1][n2] = self.PT_cap_PT_cap_EPT(n1, n2, theta)
                n1, n2 = v, u
                if n1 not in self.PTcapPTcapEPT:
                    self.PTcapPTcapEPT[n1] = {}
                    self.PTcapPTcapEPT[n1][n2] = self.PTcapPTcapEPT[n2][n1]
                else:
                    if n2 not in self.PTcapPTcapEPT[n1]:
                        self.PTcapPTcapEPT[n1][n2] = self.PTcapPTcapEPT[n2][n1]

                if len(self.PTcapPTcapEPT[temp1][temp2]) == 0:
                    EQ.add((temp1, temp2))

            while (len(EQ) > 0):
                temp1, temp2 = EQ.pop()
                ED.add((temp1, temp2))
                self.UpdatePeriodAndEPT(temp1, temp2, k)
                self.UpdatePeriodAndEPT(temp2, temp1, k)
                for temp in [temp1, temp2]:
                    for x in self.adjset[temp]:
                        if self.degree[x] >= k:
                            if x < temp:
                                u1, u2 = x, temp
                            else:
                                u1, u2 = temp, x
                            if (u1, u2) not in ED and (u1, u2) not in EQ:
                                n1, n2 = u1, u2
                                if n1 not in self.PTcapPTcapEPT:
                                    self.PTcapPTcapEPT[n1] = {}
                                    self.PTcapPTcapEPT[n1][n2] = self.PT_cap_PT_cap_EPT(n1, n2, theta)
                                else:
                                    if n2 not in self.PTcapPTcapEPT[n1]:
                                        self.PTcapPTcapEPT[n1][n2] = self.PT_cap_PT_cap_EPT(n1, n2, theta)
                                n1, n2 = u2, u1
                                if n1 not in self.PTcapPTcapEPT:
                                    self.PTcapPTcapEPT[n1] = {}
                                    self.PTcapPTcapEPT[n1][n2] = self.PTcapPTcapEPT[n2][n1]
                                else:
                                    if n2 not in self.PTcapPTcapEPT[n1]:
                                        self.PTcapPTcapEPT[n1][n2] = self.PTcapPTcapEPT[n2][n1]

                                if len(self.PTcapPTcapEPT[u1][u2]) == 0:
                                    EQ.add((u1, u2))
        return Vc, ED

    def UpdatePeriodAndEPT(self, w, v, k):
        for t in self.tGraph[w][v]:
            if t in self.IPT[w]:
                for item in self.IPT[w][t]:
                    if item in self.PT[w]:
                        key = int((t - item[0]) / item[1])
                        self.PT[w][item][key] -= 1
                        if self.PT[w][item][key] < k:
                            del (self.PT[w][item])
                            for x in self.PTcapPTcapEPT[w]:
                                if item in self.PTcapPTcapEPT[w][x]:
                                    self.PTcapPTcapEPT[w][x].remove(item)
                                    # self.PTcapPTcapEPT[x][w].remove(item)  needn't

    def PT_cap_PT_cap_EPT(self, u, v, theta):
        if len(self.PT[u]) == 0 or len(self.PT[v]) == 0:
            return set()
        edgetime = set(self.tGraph[u][v])
        if len(self.tGraph[u][v]) < theta:
            return set()

        PTtemp = []
        if len(self.PT[u]) > len(self.PT[v]):
            tem1, tem2 = v, u
        else:
            tem1, tem2 = u, v
        for pt1 in self.PT[tem1]:
            if pt1 in self.PT[tem2]:
                PTtemp.append(pt1)

        result = set()
        if len(PTtemp) > 0:
            for PT_item in PTtemp:
                flag = 1
                for i in range(theta):
                    if i * PT_item[1] + PT_item[0] not in edgetime:
                        flag = 0
                        break
                if flag == 1:
                    result.add(PT_item)
        return result

    def PT_cap_PT_cap_EPT2(self, u, v, theta):
        if len(self.PT[u]) == 0 or len(self.PT[v]) == 0:
            return set()
        edgetime = self.tGraph[u][v]
        if len(edgetime) < theta:
            return set()
        max_time_temp = max(edgetime)

        PeriodicQ = []
        PeriodicSet = set()
        StartS = []
        for t in edgetime:
            tempQ = PeriodicQ[:]
            for S in tempQ:
                if ((t - S[0]) % S[1]) == 0:
                    if (t - S[0]) / S[1] != S[2]:
                        PeriodicQ.remove(S)
                        continue
                    S[2] += 1
                    if S[2] == theta:
                        PeriodicSet.add((S[0], S[1]))
                        PeriodicQ.remove(S)
                        continue
                if (t - S[0]) > (theta - 1) * S[1]:
                    PeriodicQ.remove(S)
            for s in StartS:
                if (t - s) * (theta - 2) <= max_time_temp - t:
                    PeriodicQ.append([s, t - s, 2])
            StartS.append(t)

        if len(PeriodicSet) == 0:
            return set()

        result = set()
        for item in PeriodicSet:
            if item in self.PT[u] and item in self.PT[v]:
                result.add(item)
        return result

    def bk(self, adjset, P, R, X, k):
        if len(P) + len(R) < k + 1:
            return
        if len(P) + len(X) == 0:
            self.maximalclique.append(R)
        else:
            for v in P:
                Pn = P & adjset[v]
                Xn = X & adjset[v]
                setv = set([v])
                Rn = R | setv
                self.bk(adjset, Pn, Rn, Xn, k)
                P = P - setv
                X = X | setv

    def bk_pivot(self, adjset, P, R, X, k):
        if len(P) + len(R) < k + 1:
            return
        if len(P) + len(X) == 0:
            self.maximalclique.append(R)
        else:
            maxindex = -1
            maxindex_len = -1
            for u in P | X:
                len_new = len(P & adjset[u])
                if len_new > maxindex_len:
                    maxindex_len = len_new
                    maxindex = u
                    # if len_new == len(self.adjset[u]) or len_new == len(P): # on demond
                    #     break
            adju = adjset[maxindex]
            enum = P - adju
            for v in enum:
                Pn = P & adjset[v]
                Xn = X & adjset[v]
                setv = set([v])
                Rn = R | setv
                self.bk_pivot(adjset, Pn, Rn, Xn, k)
                P = P - setv
                X = X | setv

    def MPCKC(self, theta, k):
        Vc = set(self.adjset.keys())
        newnodes = {}
        id = 0
        for u in Vc:
            self.PT[u] = self.ComputePeriodAll(theta, k, u, Vc)
            for pt in self.PT[u]:
                newnodes[(u, pt)] = id
                id += 1
        print("#new nodes:" + str(len(newnodes)))

        newadj = {}
        for u in Vc:
            for v in self.adjset[u]:
                if self.degree[v] < k:
                    continue
                if v > u:
                    edge = self.PT_cap_PT_cap_EPT(u, v, theta)
                    for pt in edge:
                        if newnodes[(u, pt)] in newadj:
                            newadj[newnodes[(u, pt)]].append(newnodes[(v, pt)])
                        else:
                            newadj[newnodes[(u, pt)]] = [newnodes[(v, pt)]]
                        if newnodes[(v, pt)] in newadj:
                            newadj[newnodes[(v, pt)]].append(newnodes[(u, pt)])
                        else:
                            newadj[newnodes[(v, pt)]] = [newnodes[(u, pt)]]
        sum_edge = 0
        for u in newadj:
            newadj[u] = set(newadj[u])
            sum_edge += len(newadj[u])
        print("#new edges:" + str(sum_edge / 2))

        self.reset()
        self.maximalclique = []
        self.bk_pivot(newadj, set(newadj.keys()), set(), set(), k)
        print("#mpc:" + str(len(self.maximalclique)))
        return newnodes, newadj

    def MPCWC(self, theta, k):
        starttime = datetime.datetime.now()
        Vc = self.WPCorePlus(theta, k)
        endtime = datetime.datetime.now()
        interval = (endtime - starttime)
        print("WPcore time:" + str(interval))
        print("WPcore #nodes:" + str(len(Vc)))

        newnodes = {}
        id = 0
        for u in Vc:
            for pt in self.PT[u]:
                newnodes[(u, pt)] = id
                id += 1
        print("#new nodes:" + str(len(newnodes)))

        newadj = {}
        for u in Vc:
            for v in self.adjset[u]:
                if self.degree[v] < k:
                    continue
                if v > u:
                    edge = self.PT_cap_PT_cap_EPT(u, v, theta)
                    for pt in edge:
                        if newnodes[(u, pt)] in newadj:
                            newadj[newnodes[(u, pt)]].append(newnodes[(v, pt)])
                        else:
                            newadj[newnodes[(u, pt)]] = [newnodes[(v, pt)]]
                        if newnodes[(v, pt)] in newadj:
                            newadj[newnodes[(v, pt)]].append(newnodes[(u, pt)])
                        else:
                            newadj[newnodes[(v, pt)]] = [newnodes[(u, pt)]]
        sum_edge = 0
        for u in newadj:
            newadj[u] = set(newadj[u])
            sum_edge += len(newadj[u])
        print("#new edges:" + str(sum_edge / 2))

        self.reset()
        self.maximalclique = []
        self.bk_pivot(newadj, set(newadj.keys()), set(), set(), k)
        print("#mpc:" + str(len(self.maximalclique)))
        return newnodes, newadj

    def MPCSC(self, theta, k):
        starttime = datetime.datetime.now()
        Vc, ED = self.SPCore(theta, k)
        endtime = datetime.datetime.now()
        interval = (endtime - starttime)
        print("WPcore #nodes:" + str(len(Vc)))
        print("SPcore time:" + str(interval))

        newnodes = {}
        id = 0
        num = 0
        for u in Vc:
            if len(self.PT[u]) > 0:
                num += 1
            for pt in self.PT[u]:
                newnodes[(u, pt)] = id
                id += 1
        print("SPcore #nodes:" + str(num))
        print("#new nodes:" + str(len(newnodes)))

        newadj = {}
        for u in Vc:
            if len(self.PT[u]) > 0:
                for v in self.adjset[u]:
                    if self.degree[v] < k:
                        continue
                    if len(self.PT[v]) > 0:
                        for pt in self.PTcapPTcapEPT[u][v]:
                            if newnodes[(u, pt)] in newadj:
                                newadj[newnodes[(u, pt)]].append(newnodes[(v, pt)])
                            else:
                                newadj[newnodes[(u, pt)]] = [newnodes[(v, pt)]]
        sum_edge = 0
        for u in newadj:
            newadj[u] = set(newadj[u])
            sum_edge += len(newadj[u])
        print("#new edges:" + str(sum_edge / 2))

        self.reset()
        self.maximalclique = []
        self.bk_pivot(newadj, set(newadj.keys()), set(), set(), k)
        print("#mpc:" + str(len(self.maximalclique)))
        return newnodes, newadj


if __name__ == '__main__':
    # path = "C:\\dataset\\enron_month"
    # path = "C:\\dataset\\wikitalk_day"
    # path = "C:\\dataset\\askubuntu_day"
    # path = "C:\\dataset\\mathoverflow_day"
    # path = "C:\\dataset\\dblp_year"
    # path = "C:\\dataset\\youtube_day"
    path = "chess_year"

    print("Dataset:" + path)
    G = Graph(path)
    theta = 3
    k = 3
    G.prepare(theta, k)

    starttime = datetime.datetime.now()
    # result = G.WPCore(theta, k)
    # result = G.WPCorePlus(theta, k)
    # result = G.SPCore(theta, k)
    # result = G.MPCKC(theta, k)
    result = G.MPCSC(theta, k)
    # result = G.MPCWC(theta, k)
    endtime = datetime.datetime.now()
    interval = (endtime - starttime)
    print("All time:" + str(interval))
    print(result[1])
    print(G.maximalclique)
