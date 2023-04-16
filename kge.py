import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def KGE(mode, nentity, nrelation, hidden_dim, gamma, embedding_range):
    if mode == 'TransE':
        entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim))
        relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
        offset_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
    elif mode == 'RotatE':
        entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim * 2))
        relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
        offset_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 2))
    elif mode == 'HAKE':
        entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim * 2))
        relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 3))
        offset_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 2))
    elif mode == 'ComplEx':
        entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim * 2))
        relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 2))
        offset_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 2))
    elif mode == 'DistMult':
        entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim))
        relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
        offset_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))

    epsilon = 2.0
    gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
    nn.init.uniform_(tensor=entity_embedding, a=-embedding_range.item(), b=embedding_range.item())
    nn.init.uniform_(tensor=relation_embedding, a=-embedding_range.item(), b=embedding_range.item())
    nn.init.uniform_(tensor=offset_embedding, a=embedding_range.item()/2, b=embedding_range.item())

    return entity_embedding, relation_embedding, offset_embedding


def KGEcalculate(mode, embedding, rembedding, embedding_range):
    if mode == 'TransE':
        result = embedding + rembedding
        return result
    elif mode == 'RotatE':
        pi = 3.14159262358979323846
        re_head, im_head = torch.chunk(embedding, 2, dim=-1)
        phase_relation = rembedding/(embedding_range.item()/pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_tail = re_head * re_relation - im_head * im_relation
        im_tail = re_head * im_relation + im_head * re_relation

        return torch.cat((re_tail, im_tail), dim=-1)

    elif mode == 'HAKE':
        pi = 3.14159262358979323846
        phase_head, mod_head = torch.chunk(embedding, 2, dim=-1)
        phase_rela, mod_rela, bias_rela = torch.chunk(rembedding, 3, dim=-1)

        phase_head = phase_head / (embedding_range.item() / pi)
        phase_rela = phase_rela / (embedding_range.item() / pi)

        phase_result = (phase_head + phase_rela)
        phase_result = phase_result * (embedding_range.item() / pi)

        mod_rela = torch.abs(mod_rela)
        bias_rela = torch.clamp(bias_rela, max=1)

        indicator = (bias_rela < -mod_rela)
        bias_rela[indicator] = -mod_rela[indicator]

        mod_result = mod_head * ((mod_rela + bias_rela)/(1-bias_rela))

        return torch.cat((phase_result, mod_result), dim=-1)

    elif mode == 'ComplEx':
        re_head, im_head = torch.chunk(embedding, 2, dim=-1)
        re_relation, im_relation = torch.chunk(rembedding, 2, dim=-1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        return torch.cat((re_score, im_score), dim=-1)

    elif mode == 'DistMult':
        return embedding * rembedding


def KGELoss(mode, embedding, target_embedding, gamma, phase_weight, modules_weight, embedding_range):
    if mode == 'TransE':
        score = embedding - target_embedding
        score = gamma - torch.norm(score, p=1, dim=-1)
        return score

    elif mode == 'RotatE':
        re_head, im_head = torch.chunk(embedding, 2, dim=-1)
        re_tail, im_tail = torch.chunk(target_embedding, 2, dim=-1)

        re_score = re_head - re_tail
        im_score = im_head - im_tail

        score = torch.cat([re_score, im_score], dim=-1)
        score = gamma - torch.norm(score, p=1, dim=-1)
        return score

    elif mode == 'HAKE':
        phase_head, mod_head = torch.chunk(embedding, 2, dim=-1)
        phase_tail, mod_tail = torch.chunk(target_embedding, 2, dim=-1)

        pi = 3.14159262358979323846
        phase_head = phase_head / (embedding_range.item() / pi)
        phase_tail = phase_tail / (embedding_range.item() / pi)

        phase_score = phase_head - phase_tail
        r_score = mod_head - mod_tail

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=-1) * phase_weight
        r_score = torch.norm(r_score, dim=-1) * modules_weight
        return gamma - (phase_score + r_score)

    elif mode == 'ComplEx':
        re_head, im_head = torch.chunk(embedding, 2, dim=-1)
        re_tail, im_tail = torch.chunk(target_embedding, 2, dim=-1)

        return torch.sum(re_tail * re_head + im_tail * im_head, dim=-1)

    elif mode == 'DistMult':
        score = embedding * target_embedding
        score = score.sum(dim=-1)
        return score


class KGEModel(nn.Module):
    def func(self, head, rel, tail, batch_type):
        ...

    def forward(self, sample, batch_type="SINGLE"):
        if batch_type == "SINGLE":
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif batch_type == "HEAD_BATCH":
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif batch_type == "TAIL_BATCH":
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        return self.func(head, relation, tail, batch_type)

    def uniformEmbd(self, args, num_entity, num_relation):
        self.args = args
        self.device = args.device
        self.pi = 3.14159262358979323846
        self.hidden_dim = args.hidden_dim
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([args.gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / args.hidden_dim]), requires_grad=False)
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

    def train_step(self, model, optimizer, train_iterator, args, warmup=False):
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        if not warmup:
            rule_score, bias = [], []
            for h, r, t in positive_sample:
                vec = self.mat[self.rule2rel[r.item()]][h].coalesce().values()
                bias.append(not ((vec != 0) & (vec != 1)).any().item())
            bias = nn.Parameter(torch.Tensor(bias), requires_grad=False).to(self.device).float()

        positive_sample = positive_sample.to(self.device)
        negative_sample = negative_sample.to(self.device)
        subsampling_weight = subsampling_weight.to(self.device)

        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)
        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        positive_score = model(positive_sample).squeeze()
        positive_score = F.logsigmoid(positive_score)

        if not warmup:
            positive_score = positive_score * bias
            negative_score = negative_score * bias

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item() / len(negative_sample),
            'negative_sample_loss': negative_sample_loss.item() / len(negative_sample),
            'loss': loss.item() / len(negative_sample)
        }

        return log

    @staticmethod
    def calcMetric(metrics, cnt, i, ranking):
        metrics[i]['MRR'] += 1.0 / ranking
        metrics[i]['MR'] += float(ranking)
        metrics[i]['HITS@1'] += 100.0 if ranking <= 1 else 0.0
        metrics[i]['HITS@3'] += 100.0 if ranking <= 3 else 0.0
        metrics[i]['HITS@10'] += 100.0 if ranking <= 10 else 0.0
        cnt[i] += 1

    @staticmethod
    def test(model, test_iterator, args, trainstep):
        model.eval()
        metrics = {'MRR': 0, 'HITS@1': 0, 'HITS@3': 0, 'HITS@10': 0, 'MR': 0}
        with torch.no_grad():
            for step in tqdm(range(len(test_iterator)), desc="Test", ncols=60, leave=False):
                positive_sample, negative_sample, filter_bias, batch_type = next(test_iterator)

                positive_sample = positive_sample.to(args.device)
                negative_sample = negative_sample.to(args.device)
                filter_bias = filter_bias.to(args.device)

                batch_size = positive_sample.size(0)

                score = model((positive_sample, negative_sample), batch_type)
                score += filter_bias

                argsort = torch.argsort(score, dim=1, descending=True)

                if batch_type == "HEAD_BATCH":
                    positive_arg = positive_sample[:, 0]
                elif batch_type == "TAIL_BATCH":
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % batch_type)

                for i in range(batch_size):
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    ranking = 1 + ranking.item()
                    metrics['MRR'] += 1.0 / ranking
                    metrics['MR'] += float(ranking)
                    metrics['HITS@1'] += 100.0 if ranking <= 1 else 0.0
                    metrics['HITS@3'] += 100.0 if ranking <= 3 else 0.0
                    metrics['HITS@10'] += 100.0 if ranking <= 10 else 0.0

        metrics = {k: v/batch_size/(step+1) for k, v in metrics.items()}
        print(("\rSTEP:{}  " + "{}:{:.4f}  "*5).format(trainstep, *[x for pair in metrics.items() for x in pair]))

        return metrics


class ModE(KGEModel):
    def __init__(self, num_entity, num_relation, args):
        super(ModE, self).__init__()
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, args.hidden_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, args.hidden_dim))
        self.uniformEmbd(args, num_entity, num_relation)

    def func(self, head, rel, tail, batch_type):
        return self.gamma.item() - torch.norm(head * rel - tail, p=1, dim=2)


class HAKE(KGEModel):
    def __init__(self, num_entity, num_relation, args, modulus_weight=1.0, phase_weight=0.5):
        super(HAKE, self).__init__()
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, args.hidden_dim * 2))
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, args.hidden_dim * 3))
        self.uniformEmbd(args, num_entity, num_relation)
        nn.init.ones_(tensor=self.relation_embedding[:, args.hidden_dim:2 * args.hidden_dim])
        nn.init.zeros_(tensor=self.relation_embedding[:, 2 * args.hidden_dim:3 * args.hidden_dim])
        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

    def func(self, head, rel, tail, batch_type):
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if batch_type == "HEAD_BATCH":
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)


class TransE(KGEModel):
    def __init__(self, num_entity, num_relation, args):
        super(TransE, self).__init__()
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, args.hidden_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, args.hidden_dim))
        self.uniformEmbd(args, num_entity, num_relation)

    def func(self, head, rel, tail, batch_type):
        if batch_type == "HEAD_BATCH":
            score = head + (rel - tail)
        else:
            score = (head + rel) - tail
        return self.gamma.item() - torch.norm(score, p=1, dim=2)


class RotatE(KGEModel):
    def __init__(self, num_entity, num_relation, args):
        super(RotatE, self).__init__()
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, args.hidden_dim * 2))
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, args.hidden_dim))
        self.uniformEmbd(args, num_entity, num_relation)

    def func(self, head, rel, tail, batch_type):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = rel/(self.embedding_range.item()/self.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if batch_type == 'HEAD_BATCH':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        return self.gamma.item() - score.sum(dim=2)


class ComplEx(KGEModel):
    def __init__(self, num_entity, num_relation, args):
        super(ComplEx, self).__init__()
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, args.hidden_dim * 2))
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, args.hidden_dim * 2))
        self.uniformEmbd(args, num_entity, num_relation)

    def func(self, head, rel, tail, batch_type):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(rel, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if batch_type == 'HEAD_BATCH':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        return self.gamma.item() - score.sum(dim=2)


class DistMult(KGEModel):
    def __init__(self, num_entity, num_relation, args):
        super(DistMult, self).__init__()
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, args.hidden_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, args.hidden_dim))
        self.uniformEmbd(args, num_entity, num_relation)

    def func(self, head, rel, tail, batch_type):
        if batch_type == 'HEAD_BATCH':
            score = head * (rel * tail)
        else:
            score = (head * rel) * tail

        return score.sum(dim=2)


class pRotatE(KGEModel):
    def __init__(self, num_entity, num_relation, args):
        super(pRotatE, self).__init__()
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, args.hidden_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, args.hidden_dim))
        self.uniformEmbd(args, num_entity, num_relation)

    def func(self, head, rel, tail, batch_type):
        phase_head = head/(self.embedding_range.item()/self.pi)
        phase_relation = rel/(self.embedding_range.item()/self.pi)
        phase_tail = tail/(self.embedding_range.item()/self.pi)

        if batch_type == 'HEAD_BATCH':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        return self.gamma.item() - score.sum(dim=2) * self.modulus


class Mode(KGEModel):
    def __init__(self, num_entity, num_relation, args):
        super(Mode, self).__init__()
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, args.hidden_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, args.hidden_dim))
        self.uniformEmbd(args, num_entity, num_relation)

    def func(self, head, rel, tail, batch_type):
        pass


KGES = {"TransE": TransE, "RotatE": RotatE, "ComplEx": ComplEx, "DistMult": DistMult,
        "pRotatE": pRotatE, "ModE": ModE, "HAKE": HAKE}
