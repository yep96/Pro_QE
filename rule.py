import time
import random
import pickle
import torch
from tqdm import tqdm
from collections import defaultdict


class GraphRule:
    def __init__(self, rule_len, rule_thr, dataset):
        super().__init__()
        self.data = dataset
        self.rule_len = rule_len
        self.rule_thr = rule_thr
        self.device = torch.device('cpu')
        self.entity_num, self.rel_num = self.data.getinfo()
        self.id2e = self.data.id2e
        self.id2r = self.data.id2r
        self.nx = self.data.nx

        self.mat1 = [x.clone().to(self.device).to_dense().to_sparse() for x in self.data.rel_mat]
        self.mat2 = []

        self.rule_set = {}

        start_time = time.time()
        while time.time()-start_time<60 and self.sampleRules():
            pass

        for rule in list(self.rule_set.keys()):
            self.rule_set[tuple([self.negRel(x) for x in rule[:-1][::-1]] + [self.negRel(rule[-1])])] = -1

        self.addPath = 1
        self.allAddPath = 0

        self.bar_format = '{desc}{percentage:3.0f}-{total_fmt}|{bar}|[{elapsed}<{remaining}{postfix}]'

    @staticmethod
    def negRel(rel):
        if rel&1: return rel-1
        return rel+1

    def sampleRules(self):
        def rand(lis):
            if len(lis) == 0:
                return False
            if len(lis) == 1:
                return lis[0]
            return lis[random.randint(0, len(lis)-1)]
        cnt = 0
        sam = 0
        while sam < 100:
            rule = []
            node = [rand(list(self.id2e.keys()))]
            for _ in range(self.rule_len):
                next = rand(list(self.nx[node[-1]].items()))
                if next == False:
                    break
                node.append(next[0])
                rule.append(rand(next[1]))

                if node[-1] in self.nx[node[0]]:
                    tmp = self.nx[node[0]][node[-1]]
                    if (len(rule) == 1 and len(tmp) > 1) or (len(rule) > 1 and tmp):
                        sam += 1
                        head = rand(tmp)
                        if head & 1:
                            rule = tuple([self.negRel(x) for x in rule[::-1]] + [self.negRel(head)])
                        else:
                            rule = tuple(rule + [head])
                        if rule not in self.rule_set:
                            self.rule_set[rule] = -1
                            cnt += 1
                        break
        return cnt > 5

    def ifChange(self):

        if self.allAddPath:
            self.mat1 = self.mat2
        return self.addPath > 0

    def qCalConf(self, mat, rule_set):

        allCon = []

        calcCache = [[None] for _ in range(self.rule_len)]
        self.calPath  = defaultdict(list)
        self.calPath_09 = defaultdict(list)
        self.calPath_08 = defaultdict(list)
        self.calPath_07 = defaultdict(list)
        self.calPath_06 = defaultdict(list)

        for rule in tqdm(rule_set, ncols=60, bar_format=self.bar_format):
            if 0 == rule_set[rule]:
                continue
            result = mat[rule[0]].coalesce()
            for i in range(1, len(rule)-1):
                if calcCache[i][0] == rule[:i+1]:
                    result = calcCache[i][1]
                else:
                    result = torch.sparse.mm(result, mat[rule[i]]).coalesce().to_dense().to_sparse()
                    calcCache[i] = [rule[:i+1], result]

            num_result = sum(result.values().bool())
            if num_result > 0:
                num_true = result * mat[rule[-1]].bool().float()
                num_true = num_true.values()
                num_true[num_true > 1] = 1
                num_true = sum(num_true)
                conf = num_true / num_result
                rule_set[rule] = conf
                if 1 <= rule_set[rule]:
                    continue

                if rule_set[rule] > 0.9:
                    self.calPath_09[rule[-1]].append((rule, result.bool().float()))
                if rule_set[rule] > 0.8:
                    self.calPath_08[rule[-1]].append((rule, result.bool().float()))
                if rule_set[rule] > 0.7:
                    self.calPath_07[rule[-1]].append((rule, result.bool().float()))
                if rule_set[rule] > 0.6:
                    self.calPath_06[rule[-1]].append((rule, result.bool().float()))
                allCon.append(conf)
            else:
                allCon.append(0)
        allCon = sorted(allCon, key=lambda x:-x)
        length = len(allCon)
        return rule_set

    def updateMaxMat_bak(self, mat, rule_set):
        ori_cnt = sum([x.values().bool().sum() for x in mat]).item()
        mat2 = [x.clone() for x in mat]
        cnt_all = 0
        for headRule, results in tqdm(rule_set.items(), ncols=60, bar_format=self.bar_format):

            results = sorted(results, key=lambda x: -self.rule_set[x[0]])
            for rule, result in results:

                tmp = (result - result * mat2[headRule].bool().float()).coalesce()
                mat2[headRule] = (mat2[headRule] + self.rule_set[rule] * tmp).to_dense().to_sparse().coalesce()
                cnt_all += int(sum(tmp.values()))
        return mat2, cnt_all

    def updateMaxMat(self, mat, rule_set0):
        ori_cnt = sum([x.values().bool().sum() for x in mat]).item()
        mat2 = [x.clone() for x in mat]
        cnt_all = 0
        for i, rule_set in enumerate([self.calPath_09, self.calPath_08, self.calPath_07, self.calPath_06]):
            for headRule, results in tqdm(rule_set.items(), ncols=60, bar_format=self.bar_format):

                results = sorted(results, key=lambda x: -self.rule_set[x[0]])
                for rule, result in results:

                    tmp = (result - result * mat2[headRule].bool().float()).coalesce()
                    mat2[headRule] = (mat2[headRule] + self.rule_set[rule] * tmp).to_dense().to_sparse().coalesce()
                    cnt_all += int(sum(tmp.values()))
            with open(f'data/FB15k-237-betae/RuleAddedMat_{0.9-i/10:.2f}.pkl', 'wb') as data:
                pickle.dump(mat, data)
        return mat2, cnt_all

    def runEpoch(self):
        self.rule_set = self.qCalConf(self.mat1, self.rule_set)
        self.mat2, self.addPath = self.updateMaxMat(self.mat1, self.calPath)
        self.allAddPath += self.addPath

