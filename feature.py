import os, torch, tqdm

path = 'data/NELL-ind/'

with open(os.path.join(path, 'entities_emerge.txt')) as f:
    unseen = torch.tensor(sorted((int(x) for x in f.readlines())))

with open(os.path.join(path, 'entities_train.txt')) as f:
    seen = torch.tensor(sorted((int(x) for x in f.readlines())))

with open(os.path.join(path, 'stats.txt')) as f:
    nent, nrel = [int(x.strip().split(' ')[1]) for x in f.readlines()]

link_ent = torch.zeros(nent, nent)
link_rel = torch.zeros(nent, nrel*2)

print('aaaa')

with open(os.path.join(path, 'triplets_indexified.txt')) as f:
    for line in tqdm.tqdm(f, ncols=60, total=620232):
        h,r,t = [int(x) for x in line.strip().split()]
        link_ent[h,t] = link_ent[t,h] = 1
        link_rel[h,r] = link_rel[t,r+nrel] = 1

link_ent = link_ent
link_rel = link_rel


maps = torch.arange(nent)

link_ent = link_ent[:, seen]

sco = (link_ent[unseen] @ link_ent[seen].t()) + (link_rel[unseen] @ link_rel[seen].t())

maps[unseen] = seen[sco.squeeze().topk(1, dim=-1).indices.squeeze()]

torch.save(maps.cpu(), 'maps2')