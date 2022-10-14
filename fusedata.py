from collections import defaultdict
import pickle
import click
import random
import os

@click.command()
@click.option('--dataset', default="FB15k-237")
def main(dataset):
    if not os.path.exists(f'data/{dataset}-ind'):
        os.mkdir(f'data/{dataset}-ind')
    query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']

    train_queries = defaultdict(set)
    train_answers = defaultdict(set)

    test_ee_queries = defaultdict(set)
    test_ee_answers = defaultdict(set)
    test_ee_easy_answers = defaultdict(set)
    valid_ee_queries = defaultdict(set)
    valid_ee_answers = defaultdict(set)
    valid_ee_easy_answers = defaultdict(set)

    test_es_queries = defaultdict(set)
    test_es_answers = defaultdict(set)
    test_es_easy_answers = defaultdict(set)
    valid_es_queries = defaultdict(set)
    valid_es_answers = defaultdict(set)
    valid_es_easy_answers = defaultdict(set)

    test_se_queries = defaultdict(set)
    test_se_answers = defaultdict(set)
    test_se_easy_answers = defaultdict(set)
    valid_se_queries = defaultdict(set)
    valid_se_answers = defaultdict(set)
    valid_se_easy_answers = defaultdict(set)

    for query_name in query_names:
        train_query = pickle.load(open('data/%s/query/train-%s-queries.pkl'%(dataset, query_name), 'rb'))
        train_answer = pickle.load(open('data/%s/query/train-%s-answers.pkl'%(dataset, query_name), 'rb'))

        test_ee_query = pickle.load(open('data/%s/query/test-%s-ee-queries.pkl'%(dataset, query_name), 'rb'))
        test_ee_answer = pickle.load(open('data/%s/query/test-%s-ee-answers.pkl'%(dataset, query_name), 'rb'))
        test_ee_easy_answer = pickle.load(open('data/%s/query/test-%s-ee-easy-answers.pkl'%(dataset, query_name), 'rb'))

        test_es_query = pickle.load(open('data/%s/query/test-%s-es-queries.pkl'%(dataset, query_name), 'rb'))
        test_es_answer = pickle.load(open('data/%s/query/test-%s-es-answers.pkl'%(dataset, query_name), 'rb'))
        test_es_easy_answer = pickle.load(open('data/%s/query/test-%s-es-easy-answers.pkl'%(dataset, query_name), 'rb'))

        test_se_query = pickle.load(open('data/%s/query/test-%s-se-queries.pkl'%(dataset, query_name), 'rb'))
        test_se_answer = pickle.load(open('data/%s/query/test-%s-se-answers.pkl'%(dataset, query_name), 'rb'))
        test_se_easy_answer = pickle.load(open('data/%s/query/test-%s-se-easy-answers.pkl'%(dataset, query_name), 'rb'))

        KEY = list(train_query.keys())[0]

        train_queries[KEY] = train_query[KEY]
        for q in train_queries[KEY]:
            train_answers[q] = train_answer[q]

        valid_test_ee = list(test_ee_query[KEY])
        random.shuffle(valid_test_ee)
        valid_ee_queries[KEY] = set(valid_test_ee[:len(valid_test_ee)//2])
        test_ee_queries[KEY] = set(valid_test_ee[len(valid_test_ee)//2:])
        for q in valid_ee_queries[KEY]:
            valid_ee_answers[q] = test_ee_answer[q]
        for q in test_ee_queries[KEY]:
            test_ee_answers[q] = test_ee_answer[q]

        valid_test_es = list(test_es_query[KEY])
        random.shuffle(valid_test_es)
        valid_es_queries[KEY] = set(valid_test_es[:len(valid_test_es)//2])
        test_es_queries[KEY] = set(valid_test_es[len(valid_test_es)//2:])
        for q in valid_es_queries[KEY]:
            valid_es_answers[q] = test_es_answer[q]
        for q in test_es_queries[KEY]:
            test_es_answers[q] = test_es_answer[q]

        valid_test_se = list(test_se_query[KEY])
        random.shuffle(valid_test_se)
        valid_se_queries[KEY] = set(valid_test_se[:len(valid_test_se)//2])
        test_se_queries[KEY] = set(valid_test_se[len(valid_test_se)//2:])
        for q in valid_se_queries[KEY]:
            valid_se_answers[q] = test_se_answer[q]
            valid_se_easy_answers[q] = test_se_easy_answer[q]
        for q in test_se_queries[KEY]:
            test_se_answers[q] = test_se_answer[q]
            test_se_easy_answers[q] = test_se_easy_answer[q]

    for save in ['train_queries', 'train_answers', 'test_ee_queries', 'test_ee_answers', 'valid_ee_queries', 'valid_ee_answers', 'test_ee_easy_answers', 'valid_ee_easy_answers', 'test_es_queries', 'test_es_answers', 'valid_es_queries', 'valid_es_answers', 'test_es_easy_answers', 'valid_es_easy_answers', 'test_se_queries', 'test_se_answers', 'valid_se_queries', 'valid_se_answers', 'test_se_easy_answers', 'valid_se_easy_answers']:
        with open(f'data/{dataset}-ind/{save.replace("_", "-")}.pkl', 'wb') as f:
            pickle.dump(eval(save), f)

if __name__ == '__main__':
    main()
