import pickle
import os.path as osp
from secrets import choice
import numpy as np
import click
from collections import defaultdict
import random
from copy import deepcopy
import time
import pdb
import logging
import os

def set_logger(save_path, query_name, print_on_screen=False):
    log_file = os.path.join(save_path, '%s.log'%(query_name))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def index_dataset(dataset_name, emerge_ratio, force=False):
    print('Indexing dataset {0}'.format(dataset_name))
    base_path = 'data/{0}/'.format(dataset_name)
    files = ['triplets.txt']
    indexified_files = ['triplets_indexified.txt']
    return_flag = True
    for i in range(len(indexified_files)):
        if not osp.exists(osp.join(base_path, indexified_files[i])):
            return_flag = False
            break
    if return_flag and not force:
        print ("index file exists")
        return

    ent2id, rel2id, id2rel, id2ent = {}, {}, {}, {}

    entid, relid = 0, 0

    with open(osp.join(base_path, files[0])) as f:
        lines = f.readlines()
        file_len = len(lines)

    for p, indexified_p in zip(files, indexified_files):
        fw = open(osp.join(base_path, indexified_p), "w")
        with open(osp.join(base_path, p), 'r') as f:
            for i, line in enumerate(f):
                print ('[%d/%d]'%(i, file_len), end='\r')
                e1, rel, e2 = line.split('\t')
                e1 = e1.strip()
                e2 = e2.strip()
                rel = rel.strip()
                rel_reverse = '-' + rel
                rel = '+' + rel

                if p == "triplets.txt":
                    if e1 not in ent2id.keys():
                        ent2id[e1] = entid
                        id2ent[entid] = e1
                        entid += 1

                    if e2 not in ent2id.keys():
                        ent2id[e2] = entid
                        id2ent[entid] = e2
                        entid += 1

                    if not rel in rel2id.keys():
                        rel2id[rel] = relid
                        id2rel[relid] = rel
                        assert relid % 2 == 0
                        relid += 1

                    if not rel_reverse in rel2id.keys():
                        rel2id[rel_reverse] = relid
                        id2rel[relid] = rel_reverse
                        assert relid % 2 == 1
                        relid += 1

                if e1 in ent2id.keys() and e2 in ent2id.keys():
                    fw.write("\t".join([str(ent2id[e1]), str(rel2id[rel]), str(ent2id[e2])]) + "\n")
                    fw.write("\t".join([str(ent2id[e2]), str(rel2id[rel_reverse]), str(ent2id[e1])]) + "\n")
        fw.close()

    with open(osp.join(base_path, "stats.txt"), "w") as fw:
        fw.write("numentity: " + str(len(ent2id)) + "\n")
        fw.write("numrelations: " + str(len(rel2id)))
    with open(osp.join(base_path, 'ent2ind.pkl'), 'wb') as handle:
        pickle.dump(ent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'rel2ind.pkl'), 'wb') as handle:
        pickle.dump(rel2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'ind2ent.pkl'), 'wb') as handle:
        pickle.dump(id2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'ind2rel.pkl'), 'wb') as handle:
        pickle.dump(id2rel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print ('num entity: %d, num relation: %d'%(len(ent2id), len(rel2id)))
    print ("indexing finished!!")

    inductive_split(dataset_name, len(ent2id), emerge_ratio)

def inductive_split(dataset, ent_num, emerge_ratio):
    print('Split inductive dataset {0}'.format(dataset))
    base_path = 'data/{0}/'.format(dataset)

    ent_train = random.sample(range(ent_num), int(ent_num*emerge_ratio))
    ent_emerge = list(set(list(range(ent_num))) - set(ent_train))

    fw_ent_train = open(osp.join(base_path, 'entities_train.txt'), "w")
    for ent in ent_train:
        fw_ent_train.write(str(ent) + '\n')
    fw_ent_emerge = open(osp.join(base_path, 'entities_emerge.txt'), "w")
    for ent in ent_emerge:
        fw_ent_emerge.write(str(ent) + '\n')

    fw_ent_train.close()
    fw_ent_emerge.close()

    fw_tri_train = open(osp.join(base_path, 'triplets_train.txt'), "w")
    fw_tri_auxiliary = open(osp.join(base_path, 'triplets_auxiliary.txt'), "w")
    with open(osp.join(base_path, 'triplets_indexified.txt'), 'r') as f:
        for i, line in enumerate(f):
            e1, rel, e2 = line.split('\t')
            e1 = int(e1.strip())
            e2 = int(e2.strip())
            rel = rel.strip()

            if  e1 in ent_train and e2 in ent_train:
                fw_tri_train.write(line)
            else:
                fw_tri_auxiliary.write(line)

    fw_tri_train.close()
    fw_tri_auxiliary.close()

    print('Spliting finished!!')

def construct_graph(base_path, indexified_files):
    ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for indexified_p in indexified_files:
        with open(osp.join(base_path, indexified_p)) as f:
            for i, line in enumerate(f):
                if len(line) == 0:
                    continue
                e1, rel, e2 = line.split('\t')
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                rel = int(rel.strip())
                ent_out[e1][rel].add(e2)
                ent_in[e2][rel].add(e1)

    return ent_in, ent_out

def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

def write_links(dataset, ent_out, max_ans_num, name, induc_type, emerge_entity):
    queries = defaultdict(set)
    answers = defaultdict(set)
    answers_easy = defaultdict(set)
    num_more_answer = 0
    cnt_queries = 0
    emerge_entity = set(emerge_entity)
    for ent in ent_out:
        for rel in ent_out[ent]:
            if len(ent_out[ent][rel]) <= max_ans_num:
                if judge_emerge_type(['e', ['r']], (ent, (rel,)), ent_out[ent][rel], emerge_entity) != induc_type:
                    continue

                queries[('e', ('r',))].add((ent, (rel,)))
                if induc_type == 'se':
                    answers[(ent, (rel,))] = ent_out[ent][rel] & emerge_entity
                    answers_easy[(ent, (rel,))] = ent_out[ent][rel] - emerge_entity
                else:
                    answers[(ent, (rel,))] = ent_out[ent][rel]
                    answers_easy[(ent, (rel,))] = set()
                cnt_queries += 1
            else:
                num_more_answer += 1

    with open('./data/%s/query/%s-queries.pkl'%(dataset, name), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/query/%s-answers.pkl'%(dataset, name), 'wb') as f:
        pickle.dump(answers, f)
    with open('./data/%s/query/%s-easy-answers.pkl'%(dataset, name), 'wb') as f:
        pickle.dump(answers_easy, f)

    print ('num_more_answer', num_more_answer, 'cnt_queries', cnt_queries)

def ground_queries(dataset, query_structure, ent_in, ent_out, small_ent_in, small_ent_out, gen_num, max_ans_num, query_name, mode, ent2id, rel2id, induc_type, emerge_entity):
    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty, num_wrong_type = 0, 0, 0, 0, 0, 0, 0, 0, 0
    ans_num = []
    queries = defaultdict(set)
    answers = defaultdict(set)
    easy_answers = defaultdict(set)
    s0 = time.time()
    old_num_sampled = -1
    emerge_entity = set(emerge_entity)
    while num_sampled < gen_num:
        if num_sampled != 0:
            if num_sampled % (gen_num//100) == 0 and num_sampled != old_num_sampled:
                logging.info('%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s wrong type: %s'%(mode,
                    query_structure,
                    num_sampled, gen_num, (time.time()-s0)/num_sampled, num_try, num_repeat, num_more_answer,
                    num_broken, num_no_extra_answer, num_no_extra_negative, num_empty, num_wrong_type))
                old_num_sampled = num_sampled
        print ('%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s wrong type: %s'%(mode,
            query_structure,
            num_sampled, gen_num, (time.time()-s0)/(num_sampled+0.001), num_try, num_repeat, num_more_answer,
            num_broken, num_no_extra_answer, num_no_extra_negative, num_empty, num_wrong_type), end='\r')
        num_try += 1
        empty_query_structure = deepcopy(query_structure)
        answer = random.sample(ent_in.keys(), 1)[0]

        broken_flag = fill_query(empty_query_structure, ent_in, ent_out, answer, ent2id, rel2id)
        if broken_flag:
            num_broken += 1
            continue
        query = empty_query_structure
        answer_set = achieve_answer(query, ent_in, ent_out)
        small_answer_set = achieve_answer(query, small_ent_in, small_ent_out)
        if len(answer_set) == 0:
            num_empty += 1
            continue
        if mode != 'train':
            if judge_emerge_type(query_structure, query, answer_set, emerge_entity) != induc_type:
                num_wrong_type += 1
                continue
        if list2tuple(query) in queries[list2tuple(query_structure)]:
            num_repeat += 1
            continue
        queries[list2tuple(query_structure)].add(list2tuple(query))
        if induc_type == 'se':
            answers[list2tuple(query)] = answer_set & emerge_entity
            easy_answers[list2tuple(query)] = answer_set - emerge_entity
        else:
            answers[list2tuple(query)] = answer_set
            easy_answers[list2tuple(query)] = set()
        num_sampled += 1
        ans_num.append(len(answers[list2tuple(query)]))

    print ()
    logging.info ("{} tp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(ans_num), np.min(ans_num), np.mean(ans_num), np.std(ans_num)))

    if mode == 'train':
        name_to_save = '%s-%s'%(mode, query_name)
    if mode == 'test':
        name_to_save = '%s-%s-%s'%(mode, query_name, induc_type)

    with open('./data/%s/query/%s-queries.pkl'%(dataset, name_to_save), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/query/%s-answers.pkl'%(dataset, name_to_save), 'wb') as f:
        pickle.dump(answers, f)
    with open('./data/%s/query/%s-easy-answers.pkl'%(dataset, name_to_save), 'wb') as f:
        pickle.dump(easy_answers, f)
    return queries, answers

def generate_queries(dataset, query_structures, gen_num, max_ans_num, gen_train, gen_valid, gen_test, query_names, save_name, induc_type):
    base_path = './data/%s'%dataset
    indexified_files = ['triplets_train.txt', 'triplets_auxiliary.txt']
    train_ent_in, train_ent_out = construct_graph(base_path, indexified_files[:1])
    if gen_test:
        test_ent_in, test_ent_out = construct_graph(base_path, indexified_files[:2])
        test_only_ent_in, test_only_ent_out = construct_graph(base_path, indexified_files[1:2])

    ent2id = pickle.load(open(os.path.join(base_path, "ent2ind.pkl"), 'rb'))
    rel2id = pickle.load(open(os.path.join(base_path, "rel2ind.pkl"), 'rb'))

    emerge_entity_file_path = 'data/{0}/'.format(dataset) + 'entities_emerge.txt'
    emerge_entity = []
    with open(emerge_entity_file_path, 'r') as f:
        for i, line in enumerate(f):
            emerge_entity.append(int(line.strip()))


    assert len(query_structures) == 1
    idx = 0
    query_structure = query_structures[idx]
    query_name = query_names[idx] if save_name else str(idx)
    print ('general structure is', query_structure, "with name", query_name, 'with inductive_type', induc_type)

    name_to_save = query_name
    set_logger("./data/{}/".format(dataset), name_to_save)

    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_empty = 0, 0, 0, 0, 0, 0
    train_ans_num = []
    s0 = time.time()
    if gen_train:
        train_queries, train_answers = ground_queries(dataset, query_structure,
            train_ent_in, train_ent_out, defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set)),
            gen_num[0], max_ans_num, query_name, 'train', ent2id, rel2id, 'train', emerge_entity)
    if gen_test:
        test_queries, test_answers = ground_queries(dataset, query_structure,
            test_ent_in, test_ent_out, train_ent_in, train_ent_out, gen_num[2], max_ans_num, query_name, 'test', ent2id, rel2id, induc_type, emerge_entity)
    print ('%s queries generated with structure %s with type %s'%(gen_num, query_structure, induc_type))

def judge_emerge_type(query_structure, query, answer_set, emerge_entity):
    query_structure_element = eval('[%s]'%repr(query_structure).replace('[', '').replace(']', ''))
    anchor_node_index = [i for i,x in enumerate(query_structure_element) if x=='e']

    query_element = eval('[%s]'%repr(tuple2list(query)).replace('[', '').replace(']', ''))
    anchors = np.array(query_element)[anchor_node_index].tolist()
    answers = list(answer_set)

    if judge_emerge(anchors, emerge_entity):
        if judge_emerge(answers, emerge_entity):
            return 'ee'
        return 'es'
    else:
        if judge_emerge(answers, emerge_entity):
            return 'se'
        return 'ss'

def judge_emerge(nodes, emerge_entity):
    for i in nodes:
        if i in emerge_entity:
            return True
    return False


def fill_query(query_structure, ent_in, ent_out, answer, ent2id, rel2id):
    assert type(query_structure[-1]) == list
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        r = -1
        for i in range(len(query_structure[-1]))[::-1]:
            if query_structure[-1][i] == 'n':
                query_structure[-1][i] = -2
                continue
            found = False
            for j in range(40):
                r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                if r_tmp // 2 != r // 2 or r_tmp == r:
                    r = r_tmp
                    found = True
                    break
            if not found:
                return True
            query_structure[-1][i] = r
            answer = random.sample(ent_in[answer][r], 1)[0]
        if query_structure[0] == 'e':
            query_structure[0] = answer
        else:
            return fill_query(query_structure[0], ent_in, ent_out, answer, ent2id, rel2id)
    else:
        same_structure = defaultdict(list)
        for i in range(len(query_structure)):
            same_structure[list2tuple(query_structure[i])].append(i)
        for i in range(len(query_structure)):
            if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                assert i == len(query_structure) - 1
                query_structure[i][0] = -1
                continue
            broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, ent2id, rel2id)
            if broken_flag:
                return True
        for structure in same_structure:
            if len(same_structure[structure]) != 1:
                structure_set = set()
                for i in same_structure[structure]:
                    structure_set.add(list2tuple(query_structure[i]))
                if len(structure_set) < len(same_structure[structure]):
                    return True

def achieve_answer(query, ent_in, ent_out):
    assert type(query[-1]) == list
    all_relation_flag = True
    for ele in query[-1]:
        if (type(ele) != int) or (ele == -1):
            all_relation_flag = False
            break
    if all_relation_flag:
        if type(query[0]) == int:
            ent_set = set([query[0]])
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out)
        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                ent_set = set(range(len(ent_in))) - ent_set
            else:
                ent_set_traverse = set()
                for ent in ent_set:
                    ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                ent_set = ent_set_traverse
    else:
        ent_set = achieve_answer(query[0], ent_in, ent_out)
        union_flag = False
        if len(query[-1]) == 1 and query[-1][0] == -1:
            union_flag = True
        for i in range(1, len(query)):
            if not union_flag:
                ent_set = ent_set.intersection(achieve_answer(query[i], ent_in, ent_out))
            else:
                if i == len(query) - 1:
                    continue
                ent_set = ent_set.union(achieve_answer(query[i], ent_in, ent_out))
    return ent_set




@click.command()
@click.option('--dataset', default="FB15k-237")
@click.option('--seed', default=0)
@click.option('--emerge_ratio', default=0.9)
@click.option('--gen_train_num', default=0)
@click.option('--gen_valid_num', default=0)
@click.option('--gen_test_num', default=0)
@click.option('--max_ans_num', default=1e6)
@click.option('--reindex', is_flag=True, default=False)
@click.option('--gen_train', is_flag=True, default=False)
@click.option('--gen_valid', is_flag=True, default=False)
@click.option('--gen_test', is_flag=True, default=False)
@click.option('--gen_id', default=0)
@click.option('--save_name', is_flag=True, default=False)
@click.option('--index_only', is_flag=True, default=False)
@click.option('--induc_type', default=None) # choices=['ee', 'es', 'se'], e: emerge, s: seen
def main(dataset, seed, gen_train_num, gen_valid_num, gen_test_num, max_ans_num, reindex, gen_train, gen_valid, gen_test, gen_id, save_name, index_only, emerge_ratio, induc_type):
    train_num_dict = {'FB15k': 200000, "FB15k-237": 100000, "NELL": 65000}
    valid_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000}
    test_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000}
    if gen_train and gen_train_num == 0:
        if 'FB15k-237' in dataset:
            gen_train_num = train_num_dict["FB15k-237"]
        elif 'FB15k' in dataset:
            gen_train_num = train_num_dict["FB15k"]
        elif 'NELL' in dataset:
            gen_train_num = train_num_dict["NELL"]
        else:
            assert False, "you should set the gen_train_num when using the other dataset"
    if gen_valid and gen_valid_num == 0:
        if 'FB15k-237' in dataset:
            gen_valid_num = 5000
        elif 'FB15k' in dataset:
            gen_valid_num = 8000
        elif 'NELL' in dataset:
            gen_valid_num = 4000
        else:
            assert False, "you should set the gen_train_num when using the other dataset"
    if gen_test and gen_test_num == 0:
        if 'FB15k-237' in dataset:
            gen_test_num = test_num_dict["FB15k-237"]
        elif 'FB15k' in dataset:
            gen_test_num = test_num_dict["FB15k"]
        elif 'NELL' in dataset:
            gen_test_num = test_num_dict["NELL"]
        else:
            assert False, "you should set the gen_train_num when using the other dataset"
    if index_only:
        index_dataset(dataset, emerge_ratio, reindex)
        exit()
    if not os.path.exists(os.path.join('data', dataset, 'query')):
        os.mkdir(os.path.join('data', dataset, 'query'))

    e = 'e'
    r = 'r'
    n = 'n'
    u = 'u'


    query_structures = [
                        [e, [r]],
                        [e, [r, r]],
                        [e, [r, r, r]],
                        [[e, [r]], [e, [r]]],
                        [[e, [r]], [e, [r]], [e, [r]]],
                        [[e, [r, r]], [e, [r]]],
                        [[[e, [r]], [e, [r]]], [r]],
                        # negation
                        [[e, [r]], [e, [r, n]]],
                        [[e, [r]], [e, [r]], [e, [r, n]]],
                        [[e, [r, r]], [e, [r, n]]],
                        [[e, [r, r, n]], [e, [r]]],
                        [[[e, [r]], [e, [r, n]]], [r]],
                        # union
                        [[e, [r]], [e, [r]], [u]],
                        [[[e, [r]], [e, [r]], [u]], [r]]
                       ]
    query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']

    generate_queries(dataset, query_structures[gen_id:gen_id+1], [gen_train_num, gen_valid_num, gen_test_num], max_ans_num, gen_train, gen_valid, gen_test, query_names[gen_id:gen_id+1], save_name, induc_type)


if __name__ == '__main__':
    main()