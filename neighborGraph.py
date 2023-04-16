import os
import pickle
from collections import defaultdict
import random

class neighborGraph:
    def __init__(self, args) -> None:
        path = '%s'%args.data_path


        with open('%s/stats.txt'%path) as f:
            entrel = f.readlines()
            nentity = int(entrel[0].split(' ')[-1])
            nrelation = int(entrel[1].split(' ')[-1])

        emerge_entity = []
        with open(os.path.join(path, "entities_emerge.txt")) as entity_file:
            for i, line in enumerate(entity_file):
                entity = line.strip()
                emerge_entity.append(entity)
        entity_file.close()
        
        graph_entity = defaultdict(list)
        graph_relation = defaultdict(list)

        with open(os.path.join(path, "entities_emerge.txt")) as entity_file:
            for i, line in enumerate(entity_file):
                entity = line.strip()
                graph_entity[int(entity)] = []
                graph_relation[int(entity)] = []
        entity_file.close()
        with open(os.path.join(path, "entities_train.txt")) as entity_file:
            for i, line in enumerate(entity_file):
                entity = line.strip()
                graph_entity[int(entity)] = []
                graph_relation[int(entity)] = []
        entity_file.close()

        shuffle_triplet = []
        with open(os.path.join(path, "triplets_indexified.txt")) as triplets:
            for line in triplets:
                shuffle_triplet.append(line)
        triplets.close()
        random.shuffle(shuffle_triplet)

        for i, line in enumerate(shuffle_triplet):
            h, r, t = line.strip().split('\t')
            if t not in emerge_entity:
                graph_relation[int(h)].append(int(r))
                graph_entity[int(h)].append(int(t))

        for key in graph_entity:
            if len(graph_entity[key]) < args.max_neighbor:
                graph_entity[key] = graph_entity[key] + [nentity]*(args.max_neighbor-len(graph_entity[key]))
                graph_relation[key] = graph_relation[key] + [nrelation]*(args.max_neighbor-len(graph_relation[key]))
            else:
                graph_entity[key] = graph_entity[key][0:args.max_neighbor]
                graph_relation[key] = graph_relation[key][0:args.max_neighbor]

        self.graph = {}
        for key in graph_entity:
            self.graph[key] = (graph_relation[key], graph_entity[key])