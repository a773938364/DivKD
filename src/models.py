# coding: utf-8
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import RobertaTokenizerFast, RobertaModel


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill(num_mask.bool(), -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


def Mask(inputs, seq_len=None, way='multiple'):
    # seq_len is list [len, len, ...]
    # inputs is tensor , B * S * H
    if seq_len is None:
        return inputs

    if way == 'multiple':
        value = 0
    else:
        value = -1e12

    batch_size = inputs.size(0)
    max_len = inputs.size(1)
    hidden_size = inputs.size(2)

    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]

    for b in range(batch_size):
        mask_one = [temp_1 for _ in range(max_len)]
        for i in range(seq_len[b]):
            mask_one[i] = temp_0
        masked_index.append(mask_one)

    if torch.cuda.is_available():
        masked_index = torch.LongTensor(masked_index).cuda()
    inputs_masked = inputs.masked_fill_(masked_index.bool(), value)

    return inputs_masked


class DropDense(nn.Module):
    def __init__(self, in_dim, out_dim, activation=False, bias=True, dropout=None):
        super(DropDense, self).__init__()
        self.FC = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = nn.ReLU() if activation else None
        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x, x_len=None, way='multiple'):
        if self.dropout is not None:
            x = self.dropout(x)

        y = self.FC(x)

        if self.activation is not None:
            y = self.activation(y)

        if x_len is not None:
            y = Mask(y, x_len, way)

        return y


class MultiDimEncodingSentence(nn.Module):
    def __init__(self, in_dim, out_dim=None, dropout=0.2):
        # 768, 512
        super(MultiDimEncodingSentence, self).__init__()
        if out_dim is None:
            self.out_dim = in_dim
        else:
            self.out_dim = out_dim

        self.input_h = None

        self.DDense_x = DropDense(in_dim, self.out_dim, bias=False)
        self.DDense_s2t_1 = DropDense(in_dim, in_dim, activation=True, bias=True)
        self.DDense_s2t_2_1 = DropDense(in_dim, 1, bias=True)
        self.DDense_s2t_2_2 = DropDense(in_dim, self.out_dim, bias=True)

        self.dropout_H = nn.Dropout(dropout)
        self.dropout_A = nn.Dropout(dropout)
        self.dropout_A_1 = nn.Dropout(dropout)
        self.dropout_A_2 = nn.Dropout(dropout)

    def forward(self, x, x_len, seg_mask=None):
        # x 是 bert 输入 B * C_S * H
        x_H = self.dropout_H(x)
        # in_dim, self.out_dim, bias=False, ac=False, dropout=None;
        # B * C_S * 512
        H = self.DDense_x(x_H, x_len, 'multiple')

        x_A = self.dropout_A(x)
        # in_dim, in_dim, ac=True, bias=True, dropout=None
        # B * C_S * 768
        A = self.DDense_s2t_1(x_A)
        A = self.dropout_A_1(A)
        # B * C_S * 1
        A_1 = self.DDense_s2t_2_1(A, x_len, 'addition')

        x_A_2 = self.dropout_A_2(x)
        # B * C_s * 512
        A_2 = self.DDense_s2t_2_2(x_A_2, x_len, 'addition')
        # B * C_S * 1 +  B * C_s * 512  - B * C_s * 1 = B * C_s * 512
        A = A_1 + A_2 - torch.mean(A_2, dim=-1, keepdims=True)

        A1 = F.softmax(A, dim=1)  # Attention
        sent_emb = torch.sum(A1 * H, dim=1)  # 把句子中 所有的 char 向量加 得到一个 来表示 句子 B * 512

        A = torch.exp(torch.clip(A, min=-1e12, max=10))  # B * C_s * 512 元素控制在-1e12, 10
        AH = A * H  # B * C_s * 512 ;  B * C_S * 512 ->  B * C_S * 512 attention后的Hid
        #  seg_mask 是 B * W_s * C_S  ; B * C_S * 512  ->  B * W_s * 512
        # 除 attention B * W_s * 512
        # print("seg_mask:",seg_mask.size(),"AH:",AH.size())
        word_emb = torch.matmul(seg_mask, AH) / (torch.matmul(seg_mask, A) + 1e-12)

        return sent_emb, word_emb


class PosteriorEncoderVAE(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.2):
        super(PosteriorEncoderVAE, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)

        self.out_dropout = nn.Dropout(dropout)

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z=mu+sigma*epsilon
        """
        sigma = torch.exp(log_var * 0.5)
        # eps=torch.randn_like(sigma)
        eps = torch.normal(0, sigma)
        return mu + sigma * eps

    def forward(self, problem_output, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        pade_hidden = hidden
        outputs, hidden = self.gru(embedded, hidden)
        # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs

        # output = outputs[-1, :, :self.hidden_size] + outputs[0, :, self.hidden_size:]
        output = torch.mean(outputs, dim=0)

        # print(outputs.size(),output.size(),problem_output.size())
        output = torch.cat([problem_output, output], dim=1)
        # output=problem_output

        mu = self.fc1(output)
        log_var = self.fc2(output)
        z = self.reparameterization(mu, log_var)

        z = problem_output + z
        z = self.out_dropout(z)

        # S x B x H
        return outputs, problem_output, z, mu, log_var


class EncoderChar(nn.Module):
    def __init__(self, bert_path, bert_size, hidden_size, get_word_and_sent=False):
        super(EncoderChar, self).__init__()
        print('-' * 20, '调用 bert 版本如下 :', '-' * 20)
        print(bert_path.split('/')[-2:-1])
        self.tokenizer = RobertaTokenizerFast.from_pretrained(bert_path)
        self.model = RobertaModel.from_pretrained(bert_path)
        self.flag = 0
        for param in self.model.parameters():
            param.requires_grad = True
        self.trans_word = get_word_and_sent
        if self.trans_word:
            self.small_fc = nn.Linear(4 * bert_size, bert_size)
            self.md_softmax = MultiDimEncodingSentence(bert_size, hidden_size)

    def forward(self, inputs, char_len, matrix, out_all_hidden=False):
        input_all = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.model.device)
        # print("input_all:",input_all['input_ids'].size())
        # print(inputs)
        # input就是 ' '.join(ori_seg)
        # return
        output = self.model(**input_all, output_hidden_states=out_all_hidden)
        if self.trans_word and out_all_hidden is False:
            sent_emb, word_emb = self.md_softmax(output.last_hidden_states, char_len, matrix)
            return word_emb, sent_emb
        elif self.trans_word and out_all_hidden:
            if self.flag == 0:
                # print('   ***************************输出所有的潜在向量，并且做优化******************************   ')
                self.flag += 1
            all_hidden = output.hidden_states[1:]
            concatenate_pooling = torch.cat(
                (all_hidden[-1], all_hidden[-2], all_hidden[-3], all_hidden[-4]), -1
            )
            o = self.small_fc(concatenate_pooling)
            sent_emb, word_emb = self.md_softmax(o, char_len, matrix)
            return word_emb, sent_emb
        else:
            return output.last_hidden_states, output.pooler_output


class EncoderSeq_ro(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq_ro, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.Dropout = nn.Dropout(dropout)

        self.gru_pade = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

    def forward(self, word_emb, input_lengths, batch_graph, hidden=None):
        H = word_emb.transpose(0, 1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(H, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)  # S x B x 2H
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        pade_outputs = pade_outputs.transpose(0, 1)
        return pade_outputs, problem_output


class EncoderSeqVAE(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeqVAE, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.gru_pade = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.out_dropout = nn.Dropout(dropout)

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z=mu+sigma*epsilon
        """
        sigma = torch.exp(log_var * 0.5)
        # eps=torch.randn_like(sigma)
        eps = torch.normal(0, sigma)
        return mu + sigma * eps

    def forward(self, word_emb, input_lengths, batch_graph, hidden=None):
        H = word_emb.transpose(0, 1)
        # print(H.size(),len(input_lengths),input_lengths)
        packed = torch.nn.utils.rnn.pack_padded_sequence(H, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)  # S x B x 2H
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        pade_outputs = pade_outputs.transpose(0, 1)

        # vae
        mu = self.fc1(problem_output)
        log_var = self.fc2(problem_output)
        z = self.reparameterization(mu, log_var)

        z = problem_output + z
        z = self.out_dropout(z)

        return pade_outputs, problem_output, z, mu, log_var


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


# Graph Module
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# Graph_Conv
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print(input.shape)
        # print(self.weight.shape)
        support = torch.matmul(input, self.weight)
        # print(adj.shape)
        # print(support.shape)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# GCN
class GCN(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        - adjacency matrix (batch_size, K, K)
        ## Returns:
        - gcn_enhance_feature (batch_size, K, out_feat_dim)
        '''
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Graph_Module(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.5):
        super(Graph_Module, self).__init__()
        '''
        ## Variables:
        - indim: dimensionality of input node features
        - hiddim: dimensionality of the joint hidden embedding
        - outdim: dimensionality of the output node features
        - combined_feature_dim: dimensionality of the joint hidden embedding for graph
        - K: number of graph nodes/objects on the image
        '''
        self.in_dim = indim
        # self.combined_dim = outdim

        # self.edge_layer_1 = nn.Linear(indim, outdim)
        # self.edge_layer_2 = nn.Linear(outdim, outdim)

        # self.dropout = nn.Dropout(p=dropout)
        # self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        # self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)
        self.h = 4
        self.d_k = outdim // self.h

        # layer = GCN(indim, hiddim, self.d_k, dropout)
        self.graph = clones(GCN(indim, hiddim, self.d_k, dropout), 4)

        # self.Graph_0 = GCN(indim, hiddim, outdim//4, dropout)
        # self.Graph_1 = GCN(indim, hiddim, outdim//4, dropout)
        # self.Graph_2 = GCN(indim, hiddim, outdim//4, dropout)
        # self.Graph_3 = GCN(indim, hiddim, outdim//4, dropout)

        self.feed_foward = PositionwiseFeedForward(indim, hiddim, outdim, dropout)
        self.norm = LayerNorm(outdim)

    def get_adj(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''
        self.K = graph_nodes.size(1)
        graph_nodes = graph_nodes.contiguous().view(-1, self.in_dim)

        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)

        # layer 2
        h = self.edge_layer_2(h)
        h = F.relu(h)

        # outer product
        h = h.view(-1, self.K, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))

        adjacency_matrix = self.b_normal(adjacency_matrix)

        return adjacency_matrix

    def normalize(self, A, symmetric=True):
        '''
        ## Inputs:
        - adjacency matrix (K, K) : A
        ## Returns:
        - adjacency matrix (K, K)
        '''
        A = A + torch.eye(A.size(0)).cuda().float()
        d = A.sum(1)
        if symmetric:
            # D = D^{-1/2}
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else:
            D = torch.diag(torch.pow(d, -1))
            return D.mm(A)

    def b_normal(self, adj):
        batch = adj.size(0)
        for i in range(batch):
            adj[i] = self.normalize(adj[i])
        return adj

    def forward(self, graph_nodes, graph):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - graph_encode_features (batch_size, K, out_feat_dim)
        '''
        nbatches = graph_nodes.size(0)
        mbatches = graph.size(0)
        if nbatches != mbatches:
            graph_nodes = graph_nodes.transpose(0, 1)
        # adj (batch_size, K, K): adjacency matrix
        if not bool(graph.numel()):
            adj = self.get_adj(graph_nodes)
            # adj = adj.unsqueeze(1)
            # adj = torch.cat((adj,adj,adj),1)
            adj_list = [adj, adj, adj, adj]
        else:
            adj = graph.float()
            adj_list = [adj[:, 1, :], adj[:, 1, :], adj[:, 4, :], adj[:, 4, :]]
        # print(adj)

        g_feature = \
            tuple([l(graph_nodes, x) for l, x in zip(self.graph, adj_list)])
        # g_feature_0 = self.Graph_0(graph_nodes,adj[0])
        # g_feature_1 = self.Graph_1(graph_nodes,adj[1])
        # g_feature_2 = self.Graph_2(graph_nodes,adj[2])
        # g_feature_3 = self.Graph_3(graph_nodes,adj[3])
        # print('g_feature')
        # print(type(g_feature))

        g_feature = self.norm(torch.cat(g_feature, 2)) + graph_nodes
        # print('g_feature')
        # print(g_feature.shape)

        graph_encode_features = self.feed_foward(g_feature) + g_feature

        return adj, graph_encode_features