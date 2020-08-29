import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
data_dir = 'data/'

_NEG_INF = -1e9


def sample_gumbel(input, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand_like(input)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sigmoid_sample(logits, temperature):
    """Draw a sample from the Gumbel-Sigmoid distribution"""
    y = logits + sample_gumbel(logits) - sample_gumbel(logits)
    return torch.sigmoid(y / temperature)


def gumbel_sigmoid(logits, temperature, hard=False, threshold=0.5):
    y = gumbel_sigmoid_sample(logits, temperature)
    if hard:
        y_hard = torch.gt(y, threshold).float()
        y = (y_hard - y).clone().detach() + y
    return y


def gumbel_sigmoid_infer(logits, threshold=0.5):
    y = torch.sigmoid(logits)
    y_hard = torch.gt(y, threshold).float()
    return y_hard


class Embedding(nn.Module):
    '''
    position embedding and word embedding
    '''
    def __init__(self, n_word, n_pos, input_size, pos_size, position=False, pretrain=True):
        super(Embedding, self).__init__()
        self.n_word = n_word
        self.n_pos = n_pos
        self.input_size = input_size
        self.pos_size = pos_size
        self.position = position
        self.pretrain = pretrain
        self.embedding = nn.Embedding(n_word+2, input_size, padding_idx=n_word+1)
        if pretrain:
            self.embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.load(data_dir+'word_embed.npy'), dtype=np.float32)))
        if position:
            self.pos1_embedding = nn.Embedding(n_pos+1, pos_size, padding_idx=n_pos)
            self.pos1_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(91).uniform(low=-0.1, high=0.1, size=(n_pos+1, pos_size)), dtype=np.float32)))
            self.pos2_embedding = nn.Embedding(n_pos+1, pos_size, padding_idx=n_pos)
            self.pos2_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(92).uniform(low=-0.1, high=0.1, size=(n_pos+1, pos_size)), dtype=np.float32)))

    def forward(self, inputs, pos1, pos2):
        embedded = self.embedding(inputs)
        if self.position:
            p1_embed = self.pos1_embedding(pos1)
            p2_embed = self.pos2_embedding(pos2)
            embedded = torch.cat([embedded, p1_embed, p2_embed], 2)
        return embedded


class CNNEncoder(nn.Module):
    '''
    CNN sentence encoder
    '''
    def __init__(self, n_word, n_pos, input_size, pos_size, hidden_size, dropout=0.5, window=3, position=False, pretrain=True):
        super(CNNEncoder, self).__init__()
        self.n_word = n_word
        self.n_pos = n_pos
        self.input_size = input_size
        self.pos_size = pos_size
        self.hidden_size = hidden_size
        self.window = window
        self.dropout = nn.Dropout(p=dropout)
        self.position = position
        self.pretrain = pretrain
        self.embedding = Embedding(n_word, n_pos, input_size, pos_size, position, pretrain)
        self.conv2d = nn.Conv2d(input_size+pos_size*2, hidden_size, (1, window), padding=(0, 1))
        self.conv2d.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(31).uniform(low=-0.1, high=0.1, size=(hidden_size, input_size+pos_size*2, 1, window)), dtype=np.float32)))
        self.conv2d.bias = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(hidden_size), dtype=np.float32)))

    def forward(self, inputs, musk, pos1, pos2):
        embedded = self.embedding(inputs, pos1, pos2)
        embedded=embedded.transpose(1, 2).unsqueeze(2) # batch*in_channels*in_height*in_width
        conv = self.conv2d(embedded)
        conv = conv.squeeze(2).transpose(1, 2)
        pooled = torch.max(conv, dim=1)[0]
        activated = F.relu(pooled)
        output = self.dropout(activated)
        return output


class PCNNEncoder(nn.Module):
    '''
    PCNN sentence encoder
    '''
    def __init__(self, n_word, n_pos, input_size, pos_size, hidden_size, dropout=0.5, window=3, position=False, pretrain=True, max_pos=100):
        super(PCNNEncoder, self).__init__()
        self.n_word = n_word
        self.n_pos = n_pos
        self.input_size = input_size
        self.pos_size = pos_size
        self.hidden_size = hidden_size
        self.window = window
        self.dropout = nn.Dropout(p=dropout)
        self.position = position
        self.pretrain = pretrain
        self.max_pos = max_pos

        self.embedding = Embedding(n_word, n_pos, input_size, pos_size, position, pretrain)

        self.musk_embedding = nn.Embedding(4, 3)
        self.musk_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)))
        self.musk_embedding.weight.requires_grad = False

        self.conv2d = nn.Conv2d(input_size+pos_size*2, hidden_size, (1, window), padding=(0, 1))
        self.conv2d.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(31).uniform(low=-0.1, high=0.1, size=(hidden_size, input_size+pos_size*2, 1, window)), dtype=np.float32)))
        self.conv2d.bias = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(hidden_size), dtype=np.float32)))

    def forward(self, inputs, musk, pos1, pos2):
        # inputs: bag*seq_len
        # musk: bag*seq_len
        poolsize = inputs.size()[1]
        embedded = self.embedding(inputs, pos1, pos2)
        embedded=embedded.transpose(1, 2).unsqueeze(2) # batch*in_channels*in_height*in_width
        conv = self.conv2d(embedded)
        conv = conv.squeeze(2).transpose(1, 2).unsqueeze(3)
        pooled = torch.max(conv+self.musk_embedding(musk).view(-1, poolsize, 1, 3)*self.max_pos, dim=1)[0]-self.max_pos
        activated = F.relu(pooled.view(-1, self.hidden_size * 3))
        output = self.dropout(activated)
        return output


class GRUEncoder(nn.Module):
    '''
    GRU sentence encoder
    '''
    def __init__(self, n_word, n_pos, input_size, pos_size, hidden_size, dropout=0.5, position=False, bidirectional=True, pretrain=True):
        super(GRUEncoder, self).__init__()
        self.n_word = n_word
        self.n_pos = n_pos
        self.input_size = input_size
        self.pos_size = pos_size
        self.hidden_size = hidden_size
        self.position = position
        self.bidirectional = bidirectional
        self.pretrain = pretrain
        self.embedding = Embedding(n_word, n_pos, input_size, pos_size, position, pretrain)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, inputs, musk, pos1, pos2):
        embedded = self.embedding(inputs, pos1, pos2)
        output, hidden = self.gru(embedded)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        weight = F.softmax(torch.matmul(output, self.W), 1)
        return torch.sum(weight*output, 1)


class BagEncoder(nn.Module):
    def __init__(self, n_word, n_pos, input_size, pos_size, hidden_size, position=False, encode_model='BiGRU', dropout=0.5):
        super(BagEncoder, self).__init__()
        self.n_word = n_word
        self.n_pos = n_pos
        self.input_size = input_size
        self.pos_size = pos_size
        self.hidden_size = hidden_size
        self.position = position
        self.encode_model = encode_model
        self.dropout = dropout
        if self.encode_model=='BiGRU':
            self.encoder = GRUEncoder(n_word, n_pos, input_size, pos_size, hidden_size, dropout=dropout, position=position)
        elif self.encode_model=='CNN':
            self.encoder = CNNEncoder(n_word, n_pos, input_size, pos_size, hidden_size, dropout=dropout, position=position)
        elif self.encode_model=='PCNN':
            self.encoder = PCNNEncoder(n_word, n_pos, input_size, pos_size, hidden_size, dropout=dropout, position=position)

    def forward(self, inputs, musk, pos1, pos2):
        # inputs: [bag1*seq_len, bag2*seq_len, ......]
        lens = [inputs_i.size(0) for inputs_i in inputs]
        inputs = torch.cat(inputs, 0)
        musk = torch.cat(musk, 0)
        pos1 = torch.cat(pos1, 0)
        pos2 = torch.cat(pos2, 0)
        out_puts = self.encoder(inputs, musk, pos1, pos2)
        return out_puts, lens #[bag1*h, bag2*h, ...]


class Extractor(nn.Module):
    '''
    sentence selector
    '''
    def __init__(self, input_size, n_class, reduce_method='multir_att', use_bias=True):
        super(Extractor, self).__init__()
        self.input_size = input_size
        self.n_class = n_class
        self.reduce_method = reduce_method
        self.use_bias = use_bias
        self.label_embedding = nn.Embedding(n_class, input_size)
        self.label_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(201).uniform(low=-0.1, high=0.1, size=(n_class, input_size)), dtype=np.float32)))
        self.A = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(211).uniform(low=-0.1, high=0.1, size=(input_size)), dtype=np.float32)))
        if self.use_bias:
            self.bias = nn.Parameter(torch.from_numpy(np.zeros((1, n_class), dtype=np.float32)))

    def divide_conquer(self, inputs, lens, fn):
        inputs = inputs.split(lens, 0)
        inputs = torch.cat([fn(inputs_i) for inputs_i in inputs], 0)
        return inputs

    def batchify(self, inputs, lens):
        max_len = max(lens)
        batch_size, dim = inputs.size()
        assert sum(lens) == batch_size

        index = [list(range(max_len * i, l + max_len * i)) for i, l in enumerate(lens)]
        index = sum(index, [])
        index = torch.tensor(index, dtype=torch.long, device=inputs.device)

        mask = torch.zeros((max_len * len(lens)), dtype=torch.bool, device=inputs.device)
        mask = mask.scatter(0, index, torch.ones_like(index, dtype=torch.bool))
        mask = mask.view(len(lens), max_len)

        new_inputs = torch.zeros((max_len * len(lens), inputs.size(-1)),
                                 dtype=inputs.dtype, device=inputs.device)
        index = index.unsqueeze(-1).expand_as(inputs)
        new_inputs = new_inputs.scatter(0, index, inputs).view(len(lens), max_len, -1)
        new_inputs = new_inputs.view(len(lens), max_len, -1)
        return new_inputs, mask

    # multi-relation attention
    def multir_att(self, sent_embeddings, lens):
        label_embeddings = self.label_embedding.weight
        sent_embeddings, mask = self.batchify(sent_embeddings, lens)
        batch_size, bag_size, dim = sent_embeddings.size()
        score = torch.mm(sent_embeddings.flatten(0, 1) * self.A, label_embeddings.t())
        score = score.view(batch_size, bag_size, -1)
        mask = mask.unsqueeze(-1).expand_as(score)
        score = score.masked_fill_(~mask, _NEG_INF)
        # batch_size * n_class * bag_size
        weight = F.softmax(score, 1).transpose(1, 2)
        # batch_size * n_class * dim
        reduce_bag = torch.bmm(weight, sent_embeddings)
        scores = torch.sum(reduce_bag * label_embeddings.unsqueeze(0), 2)
        if self.use_bias:
            scores = scores + self.bias
        return scores

    def mean(self, sent_embeddings, lens):
        label_embeddings = self.label_embedding.weight
        sent_embeddings, mask = self.batchify(sent_embeddings, lens)
        mask = torch.where(mask, torch.ones_like(mask, dtype=torch.float32),
                                 torch.zeros_like(mask, dtype=torch.float32))
        reduce_bag = sent_embeddings.sum(1) / mask.sum(1, keepdim=True)
        scores = torch.mm(reduce_bag, label_embeddings.t())
        if self.use_bias:
            scores = scores + self.bias
        return scores

    def cross_max(self, sent_embeddings, lens):
        label_embeddings = self.label_embedding.weight
        sent_embeddings, mask = self.batchify(sent_embeddings, lens)
        mask = mask.unsqueeze(-1).expand_as(sent_embeddings)
        sent_embeddings.masked_fill_(~mask, _NEG_INF)
        reduce_bag = sent_embeddings.max(1)[0]
        scores = torch.mm(reduce_bag, label_embeddings.t())
        if self.use_bias:
            scores = scores + self.bias
        return scores

    def forward(self, inputs, lens):
        # Sentence embedding in a bag
        sent_embeddings = inputs
        if self.reduce_method=='multir_att':
            scores = self.multir_att(sent_embeddings, lens)
        # Cross-sentence Max-musking
        elif self.reduce_method=='cross_max':
            scores = self.cross_max(sent_embeddings, lens)
        elif self.reduce_method=='mean':
            scores = self.mean(sent_embeddings, lens)
        return scores

    def pred(self, inputs):
        scores_all = []
        for i in range(len(inputs)):
            # Sentence embedding in a bag
            sent_embeddings = inputs[i]
            if self.reduce_method=='multir_att':
                scores = self.multir_att(sent_embeddings)
            # Cross-sentence Max-musking
            elif self.reduce_method=='cross_max':
                scores = self.cross_max(sent_embeddings)
            elif self.reduce_method=='mean':
                scores = self.mean(sent_embeddings)
            scores_all.append(scores)
        return torch.cat(scores_all, 0)    # batchs * dims


class Z_Y(nn.Module):
    """nEM transition module: P(Z|Y)"""
    def __init__(self, n_class, init1=1., init2=0., na_init1=1., na_init2=0., requires_grad=False, norm=False):
        # init1=p(z=1|y=1), init2=p(z=1|y=0)
        super(Z_Y, self).__init__()
        self.n_class = n_class
        self.init1 = init1
        self.init2 = init2
        self.na_init1 = na_init1
        self.na_init2 = na_init2
        self.norm = norm
        self.phi = nn.Embedding(n_class, 2)
        temp = np.tile(np.asarray([init1, init2], dtype=np.float32).reshape(1, 2), (n_class, 1))
        temp[0, 0] = na_init1
        temp[0, 1] = na_init2
        if self.norm:
            temp = np.log(temp + 1e-5) - np.log(1 - temp + 1e-5)
        self.phi.weight = nn.Parameter(torch.from_numpy(temp))
        self.phi.weight.requires_grad = requires_grad
        self.mask_z = nn.Embedding(2, 2)
        self.mask_z.weight = nn.Parameter(torch.from_numpy(np.asarray([[0., 1.],[1., 0.]], dtype=np.float32)))
        self.mask_z.weight.requires_grad = False

    def init_transition(self, cond_prob):
        if self.norm:
            cond_prob = np.log(cond_prob + 1e-5) - np.log(1 - cond_prob + 1e-5)
        self.phi.weight.data = torch.tensor(cond_prob, device=self.phi.weight.data.device)

    def forward(self, z):
        if self.norm:
            z_y = torch.sigmoid(self.phi.weight)
        else:
            z_y = self.phi.weight
        z_y = torch.clamp(z_y, 0, 1)
        # print(z_y)
        _z_y = 1. - z_y
        z_y_ = torch.cat([z_y, _z_y], 1).view(-1, 2, 2)
        musk = self.mask_z(z).unsqueeze(2)
        z_y = torch.matmul(musk, z_y_).squeeze(2)
        return z_y


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, weight = self.self_attn(src, src, src, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weight


class TransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        weights = []
        for mod in self.layers:
            output, weight = mod(output, src_mask=mask,
                                 src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)
            if self.norm is not None:
                output = self.norm(output)
        weight = torch.stack(weights).mean(0)
        return output, weight


class GlobalAttention(nn.Module):
    def __init__(self, dim, attn_type='dot', tanh_query=False):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        self.tanh_query = tanh_query
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")
        if self.attn_type == 'general':
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == 'mlp':
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, 1, bias=False)
            self.b = nn.Parameter(torch.randn(1, 1, 1, dim))

    def forward(self, h_t, h_s):
        if self.attn_type in ['general', 'dot']:
            if self.attn_type == 'general':
                h_t = self.linear_in(h_t)
                h_t = torch.tanh(h_t) if self.tanh_query else h_t
            h_s = h_s.transpose(1, 2)
            return torch.bmm(h_t, h_s)

        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        wq = self.linear_query(h_t)
        wq = torch.unsqueeze(wq, 2)
        wq = wq.expand(-1, -1, src_len, -1)
        uh = self.linear_context(h_s)
        uh = torch.unsqueeze(uh, 1)
        uh = uh.expand(-1, tgt_len, -1, -1)
        wquh = torch.tanh(wq + uh + self.b)
        return self.v(wquh).squeeze(-1)


class Z_Y_(nn.Module):
    def __init__(self, config, n_class, input_size, encode_model='PCNN', label_output_embedding=None, use_bias=True, base=False):
        super(Z_Y_, self).__init__()
        self.config = config
        self.n_class = n_class
        self.use_bias = use_bias
        self.base = base
        if encode_model == 'PCNN' and label_output_embedding is not None:
            input_size *= 3
        self.label_input_embedding_0 = nn.Embedding(n_class, input_size)
        self.label_input_embedding_1 = nn.Embedding(n_class, input_size)
        if label_output_embedding is None:
            self.label_output_embedding = nn.Embedding(n_class, input_size)
        else:
            self.label_output_embedding = label_output_embedding
        if self.use_bias:
            self.bias = nn.Parameter(torch.from_numpy(np.zeros((1, n_class), dtype=np.float32)))

        if config.transform:
            encoder_layer = TransformerEncoderLayer(input_size, config.n_head,
                                                    input_size, config.dropout)
            self.encoder = TransformerEncoder(encoder_layer, config.num_layers)
            if config.transform_norm:
                self.encoder_norm = nn.LayerNorm(input_size)
        else:
            self.encoder = GlobalAttention(input_size, config.attn_type, config.tanh_query)

    def forward(self, pred):
        if not self.base:
            pred = gumbel_sigmoid(F.logsigmoid(pred), self.config.temperature,
                                  self.config.hard, self.config.threshold)
        input_embedding_1 = self.label_input_embedding_1.weight.unsqueeze(0) * pred.unsqueeze(2)
        input_embedding_0 = self.label_input_embedding_0.weight.unsqueeze(0) * (1 - pred).unsqueeze(2)
        input_embedding = input_embedding_1 + input_embedding_0
        n_label = input_embedding.size(1)

        if self.config.transform:
            input_embedding = input_embedding.transpose(0, 1)
            if self.config.transform_norm:
                input_embedding = self.encoder_norm(input_embedding)
            if self.config.indep:
                mask = torch.eye(n_label, n_label, dtype=torch.float32, device=input_embedding.device)
                mask = (1 - mask) * _NEG_INF
            else:
                mask = None
            new_input_embedding, weight = self.encoder(input_embedding, mask=mask)
            new_input_embedding = new_input_embedding.transpose(0, 1)
        else:
            weight = self.encoder(input_embedding, input_embedding)
            if self.config.indep:
                mask = torch.eye(n_label, n_label, dtype=torch.bool, device=input_embedding.device)
                mask = ~mask
            else:
                mask = torch.zeros((n_label, n_label), dtype=torch.bool, device=input_embedding.device)
            mask = mask.unsqueeze(0)
            weight.masked_fill_(mask, _NEG_INF)
            weight = F.softmax(weight, -1)
            new_input_embedding = torch.bmm(weight, input_embedding)

        label_embeddings = self.label_output_embedding.weight.unsqueeze(0)
        scores = torch.sum(new_input_embedding * label_embeddings, 2)
        if self.use_bias:
            scors = scores + self.bias

        if self.config.test_show:
            plt.imshow(weight.sum(0).data.cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.savefig('heatmap.png')
            print('heatmap.png generated!')
            input()
        return scores, weight


class Y_S(nn.Module):
    """nEM prediction module: P(Y|S)"""
    def __init__(self, embed_dim, pos_dim, hidden_dim, n_word, n_pos, n_class, reg_weight, reduce_method='mean', position=True, encode_model='PCNN', dropout=0.5, sigmoid=False):
        super(Y_S, self).__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.n_word = n_word
        self.n_pos = n_pos
        self.n_class = n_class
        self.reg_weight = reg_weight
        self.reduce_method = reduce_method
        self.position = position
        self.encode_model = encode_model
        self.dropout = dropout
        self.sigmoid = sigmoid
        self.bag_encoder = BagEncoder(n_word, n_pos, embed_dim, pos_dim, hidden_dim, position, encode_model, dropout)
        if encode_model=='PCNN':
            hidden_dim = hidden_dim*3
        self.extractor = Extractor(hidden_dim, n_class, reduce_method)

    def forward(self, bags, musk_idxs, pos1, pos2):
        groups, lens = self.bag_encoder(bags, musk_idxs, pos1, pos2)
        scores = self.extractor(groups, lens)
        if self.sigmoid:
            y1_s = torch.sigmoid(scores)
        else:
            y1_s = F.softmax(scores, 1)
        return y1_s


class RE(nn.Module):
    """traning nEM"""
    def __init__(self, config, embed_dim, pos_dim, hidden_dim, n_word, n_pos, n_class, reg_weight, reduce_method='mean', position=True, encode_model='PCNN', dropout=0.5, sigmoid=False, init1=0., init2=0., na_init1=10., na_init2=-10., requires_grad=False, norm=False, q=0., l2=False, noise=False, beta=1., base=False):
        super(RE, self).__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.n_word = n_word
        self.n_pos = n_pos
        self.n_class = n_class
        self.reg_weight = reg_weight
        self.reduce_method = reduce_method
        self.position = position
        self.encode_model = encode_model
        self.dropout = dropout
        self.sigmoid = sigmoid
        self.init1 = init1
        self.init2 = init2
        self.na_init1 = na_init1
        self.na_init2 = na_init2
        self.q = q
        self.l2 = l2
        self.beta = beta
        self.config = config
        self.y_s = Y_S(embed_dim, pos_dim, hidden_dim, n_word, n_pos, n_class, reg_weight, reduce_method, position, encode_model, dropout, sigmoid)
        if noise:
            self.z_y = Z_Y_(config, n_class, hidden_dim, encode_model=encode_model,
                            label_output_embedding=self.y_s.extractor.label_embedding, base=base)
        else:
            self.z_y = Z_Y(n_class, init1, init2, na_init1, na_init2, requires_grad, norm)

    def reg_loss(self, pretrain=False):
        reg_c = torch.norm(self.y_s.bag_encoder.encoder.embedding.embedding.weight) + torch.norm(self.y_s.extractor.label_embedding.weight) + torch.norm(self.y_s.extractor.A)
        if self.position:
            reg_c = reg_c + torch.norm(self.y_s.bag_encoder.encoder.embedding.pos1_embedding.weight) + torch.norm(self.y_s.bag_encoder.encoder.embedding.pos2_embedding.weight)
        if self.encode_model=='BiGRU':
            reg_c = reg_c + torch.norm(self.y_s.bag_encoder.encoder.W)
        elif self.encode_model=='CNN' or self.encode_model=='PCNN':
            reg_c = reg_c + torch.norm(self.y_s.bag_encoder.encoder.conv2d.weight)
        if pretrain:
            return reg_c

        if self.config.noise and self.config.noise_wd:
            reg_c = reg_c + torch.norm(self.z_y.label_input_embedding_0.weight) + \
                            torch.norm(self.z_y.label_input_embedding_1.weight)
        if self.config.transform and self.config.transform_wd:
            for l in range(self.config.num_layers):
                reg_c = reg_c + torch.norm(getattr(self.z_y.encoder.layers, str(l)).self_attn.in_proj_weight) + \
                                torch.norm(getattr(self.z_y.encoder.layers, str(l)).self_attn.out_proj.weight) + \
                                torch.norm(getattr(self.z_y.encoder.layers, str(l)).linear1.weight) + \
                                torch.norm(getattr(self.z_y.encoder.layers, str(l)).linear2.weight)
        return reg_c

    def L_q_loss(self, pred, y, q):
        if q == 0.:
            if self.sigmoid:
                return y*torch.log(pred)+(1.-y)*torch.log(1.-pred)
            else:
                return y*torch.log(pred)
        if self.sigmoid:
            return y*(pred**q-1.)/q+(1.-y)*((1.-pred)**q-1.)/q
        return y*(pred**q-1.)/q

    def L_2_loss(self, pred, y):
        if self.sigmoid:
            return -2 * (pred - y) ** 2
        else:
            return -(pred - y) ** 2

    def baseModel(self, bags, musk_idxs, pos1, pos2, labels):
        pred = self.y_s(bags, musk_idxs, pos1, pos2)
        output = torch.max(pred, dim=1)[1]
        pred = torch.clamp(pred, 1e-4, 1.0-1e-4)
        y = labels.to(torch.float)
        if self.l2:
            sum_i = torch.sum(self.L_2_loss(pred, y), 1)
        else:
            sum_i = torch.sum(self.L_q_loss(pred, y, self.q), 1)
        loss = -torch.mean(sum_i)
        return loss+self.reg_weight*self.reg_loss(True)

    def noiseModel(self, bags, musk_idxs, pos1, pos2, labels):
        y1_s = self.y_s(bags, musk_idxs, pos1, pos2)
        logits, weight = self.z_y(y1_s)
        pred_1 = torch.clamp(y1_s, 1e-4, 1.0-1e-4)
        pred_2 = torch.sigmoid(logits)
        pred_2 = torch.clamp(pred_2, 1e-4, 1.0-1e-4)
        y = labels.to(torch.float)
        if self.l2:
            sum_i_1 = torch.sum(self.L_2_loss(pred_1, y), 1)
            sum_i_2 = torch.sum(self.L_2_loss(pred_2, y), 1)
        else:
            sum_i_1 = torch.sum(self.L_q_loss(pred_1, y, self.q), 1)
            sum_i_2 = torch.sum(self.L_q_loss(pred_2, y, self.q), 1)
        loss_1 = -torch.mean(sum_i_1)
        loss_2 = -torch.mean(sum_i_2)
        loss = loss_2 * self.beta + loss_1 * (1 - self.beta)
        if self.config.mid_reg_l2:
            loss = loss - torch.mean(self.config.mid_reg_l2 * self.L_2_loss(pred_1, y))
        if self.config.mid_reg_l1:
            loss = loss - torch.mean(self.config.mid_reg_l1 * self.L_q_loss(pred_1, y, 1))
        if self.config.norm_attn:
            weight_norm = torch.eye(weight.size(1), dtype=torch.float32, device=weight.device)
            loss = loss + torch.mean((weight - weight_norm.unsqueeze(0)) ** 2)
        return loss+self.reg_weight*self.reg_loss(False)

    def noiseModel_RL(self, bags, musk_idxs, pos1, pos2, labels):
        rl_num = self.config.rl_num
        y1_s = self.y_s(bags, musk_idxs, pos1, pos2)
        y1_s = y1_s.repeat(rl_num, 1)
        samples = torch.bernoulli(y1_s)
        log_like = torch.log(y1_s * samples + (1 - y1_s) * (1 - samples)).sum(-1)
        logits, weight = self.z_y(samples)
        pred_1 = torch.clamp(y1_s, 1e-4, 1.0-1e-4)
        pred_2 = torch.sigmoid(logits)
        pred_2 = torch.clamp(pred_2, 1e-4, 1.0-1e-4)
        y = labels.to(torch.float)
        y = y.repeat(rl_num, 1)
        if self.l2:
            sum_i_1 = torch.sum(self.L_2_loss(pred_1, y), 1)
            sum_i_2 = torch.sum(self.L_2_loss(pred_2, y), 1)
        else:
            sum_i_1 = torch.sum(self.L_q_loss(pred_1, y, self.q), 1)
            sum_i_2 = torch.sum(self.L_q_loss(pred_2, y, self.q), 1)
        loss_1 = -torch.mean(sum_i_1)
        loss_2 = -torch.mean(sum_i_2)
        loss = loss_2 * self.beta + loss_1 * (1 - self.beta)
        if self.config.mid_reg_l2:
            loss = loss - torch.mean(self.config.mid_reg_l2 * self.L_2_loss(pred_1, y))
        if self.config.mid_reg_l1:
            loss = loss - torch.mean(self.config.mid_reg_l1 * self.L_q_loss(pred_1, y, 1))
        if self.config.norm_attn:
            weight_norm = torch.eye(weight.size(1), dtype=torch.float32, device=weight.device)
            loss = loss + torch.mean((weight - weight_norm.unsqueeze(0)) ** 2)
        if self.config.rl_reward == 'mean_lik':
            reward = torch.mean(torch.log(pred_2 * y + (1 - pred_2) * (1 - y)), -1)
        elif self.config.rl_reward == 'sum_lik':
            reward = torch.sum(torch.log(pred_2 * y + (1 - pred_2) * (1 - y)), -1)
        else:
            raise NotImplementedError
        reward = reward.view(rl_num, -1)
        reward_mean = reward.mean(0, keepdim=True)
        reward = (reward - reward_mean).view(-1)
        rl_loss = -torch.mean(reward.clone().detach() * log_like)
        return loss+self.reg_weight*self.reg_loss(False)+self.config.rl_weight*rl_loss

    # e_step: computing Q(Y)
    def E_step(self, bags, musk_idxs, pos1, pos2, labels):
        y1_s = self.y_s(bags, musk_idxs, pos1, pos2)
        y0_s = 1. - y1_s
        y_s = torch.cat([y1_s.unsqueeze(2), y0_s.unsqueeze(2)], 2)
        z_y = self.z_y(labels)
        z_y_y_s = z_y*y_s
        z_y_y_s_ = torch.unbind(z_y_y_s, 2)[0]
        z_s = torch.sum(z_y_y_s, 2).clamp(min=1e-5)
        Q_y1 = z_y_y_s_/z_s
        return Q_y1

    # m_step: loss function (lower bound). optimizing usiing gradient descent
    def M_step(self, bags, musk_idxs, pos1, pos2, labels, Q_y1):
        y1_s = self.y_s(bags, musk_idxs, pos1, pos2)
        y0_s = 1. - y1_s
        y_s = torch.cat([y1_s.unsqueeze(2), y0_s.unsqueeze(2)], 2)
        y_s = torch.clamp(y_s, 1e-5, 1.0-1e-5)
        z_y = self.z_y(labels)
        z_y = torch.clamp(z_y, 1e-5, 1.0-1e-5)
        Q_y0 = 1. - Q_y1
        Q_y = torch.cat([Q_y1.unsqueeze(2), Q_y0.unsqueeze(2)], 2)
        if self.l2:
            z_y_y_s = z_y * y_s
            sum_bit = -(Q_y - z_y_y_s) ** 2
        else:
            if self.q == 0.:
                log_z_y_y_s = torch.log(z_y) + torch.log(y_s)
            else:
                z_y_y_s = z_y * y_s
                log_z_y_y_s = (z_y_y_s ** self.q - 1.) / self.q
            sum_bit = torch.sum(Q_y*log_z_y_y_s, 2)
        sum_i = torch.sum(sum_bit, 1)
        loss = -torch.mean(sum_i)
        return loss+self.reg_weight*self.reg_loss(False)

    def pred(self, bags, musk_idxs, pos1, pos2):
        return self.y_s(bags, musk_idxs, pos1, pos2)

    def pred_show(self, bags, musk_idxs, pos1, pos2):
        y1_s = self.y_s(bags, musk_idxs, pos1, pos2)
        logits, _ = self.z_y(y1_s)
        return torch.sigmoid(logits)
