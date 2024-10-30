# coding: utf-8

from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["["], word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["]"]:
                res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
                                      word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["["], word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
                        word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    nums_stack_batch=copy.deepcopy(nums_stack_batch)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target[i]==unk:
            print("target:",target,target[i],decoder_output.size())
            print("decoder_output:",decoder_output[i])
            print("num_stack:", num_stack)
            target[i]=num_start

        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)
    # print("num_stack",num_stack,num_list)

    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test, tar

    # print(test, tar)
    if test is None:
        print("predict result is None")
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    masked_index=masked_index.bool()
    return all_num.masked_fill_(masked_index, 0.0)


def train_attn(input_batch, input_length, target_batch, target_length, num_batch, nums_stack_batch, copy_nums,
               generate_nums, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang, clip=0,
               use_teacher_forcing=1, beam_size=1, english=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    # seq_mask = torch.ByteTensor(seq_mask)
    seq_mask=torch.BoolTensor(seq_mask)

    num_start = output_lang.n_words - copy_nums - 2
    unk = output_lang.word2index["UNK"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)

    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_length, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                                               num_start, copy_nums, generate_nums, english)
                if USE_CUDA:
                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    if clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return return_loss


def evaluate_attn(input_seq, input_length, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
                  beam_size=1, english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(1)
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, [input_length], None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    beam_list = list()
    score = 0
    beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == output_lang.word2index["EOS"]:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0].all_output
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.cuda()
            all_hidden = all_hidden.cuda()
        all_outputs = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == output_lang.word2index["EOS"]:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
            #                                1, num_start, copy_nums, generate_nums, english)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            score = f.log_softmax(decoder_output, dim=1)
            score += beam_list[b_idx].score
            beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            all_outputs.append(beam_list[b_idx].all_output)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output = all_outputs[indices]+[word_input]
            temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0].all_output


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

def get_encoder_decoder_outputs_vae_(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge,eq_encoder,  output_lang, num_pos,batch_graph, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)
    seq_mask=seq_mask.bool()

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)
    num_mask=num_mask.bool()

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)
    batch_graph = torch.LongTensor(batch_graph)
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph=batch_graph.cuda()


    # Run words through encoder

    encoder_outputs, problem_output,z, mu, log_var = encoder(input_var, input_length, batch_graph)

    #posterior vae
    teacher_target = copy.deepcopy(target)
    eq_encoder_outputs, problem_output_post,mu_post,log_var_post = eq_encoder(problem_output,teacher_target.cuda(),target_length)

    # problem_output=problem_output+z_post
    # problem_output = problem_output + z

    problem_output=problem_output_post


    # Prepare input and output variables
    node_stacks = [[TreeNode(root)] for root in problem_output.split(1, dim=0)]

    # max_target_length = max(target_length)
    max_target_length=target.size(0)


    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)
        # print(nums_stack_batch)
        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                if i-num_start<current_nums_embeddings.size(1):
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                else:
                    print("target:",i,i-num_start,current_nums_embeddings.size())
                    print(target[t].tolist())
                    current_num = current_nums_embeddings[idx, -1].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    return encoder_outputs, problem_output,mu,log_var,all_node_outputs,target,mu_post,log_var_post

def get_encoder_decoder_outputs_evaluate(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,bert,
               encoder, predict, generate, merge,  output_lang, num_pos,batch_graph,mat, ori_datas, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)
    seq_mask=seq_mask.bool()

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)
    num_mask=num_mask.bool()

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)
    batch_graph = torch.LongTensor(batch_graph)
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)
    mat = torch.FloatTensor(np.array(mat))

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph=batch_graph.cuda()
        mat=mat.cuda()


    # print(mat.shape[-1],mat.shape)
    # Run words through encoder
    word_emb, sent_emb = bert(' '.join(ori_datas), [mat.shape[-1]], mat,
                              out_all_hidden=True)  # B[1] * char_len_max * 512

    encoder_outputs, problem_output = encoder(word_emb, input_length, batch_graph)

    # encoder_outputs, problem_output = encoder(input_var, input_length, batch_graph)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    # max_target_length = max(target_length)
    max_target_length=target.size(0)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)
        # print(nums_stack_batch)
        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                if i-num_start<current_nums_embeddings.size(1):
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                else:
                    print("target:",i,i-num_start,current_nums_embeddings.size())
                    print(target[t].tolist())
                    current_num = current_nums_embeddings[idx, -1].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    return encoder_outputs, problem_output,all_node_outputs,target



def get_encoder_decoder_outputs_2decoder(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, predict2,generate2, merge2,  output_lang, num_pos, batch_graph,english=False):
    encoder_outputs,problem_output, all_nums_encoder_outputs=get_encoder_outputs(input_batch, input_length,encoder, num_pos,batch_graph, english=english)
    all_node_outputs1,target=get_decoder_outputs(encoder_outputs,problem_output,all_nums_encoder_outputs, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
                predict, generate, merge,  output_lang, num_pos, english=english)
    
    all_node_outputs2,target=get_decoder_outputs(encoder_outputs,problem_output,all_nums_encoder_outputs, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
             predict2, generate2, merge2,  output_lang, num_pos, english=english)
    

    return encoder_outputs, problem_output,all_node_outputs1,all_node_outputs2,target

def get_encoder_outputs(input_batch, input_length,encoder, num_pos,batch_graph, english=False):

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    batch_graph = torch.LongTensor(batch_graph)
    batch_size = len(input_length)

    if USE_CUDA:
        input_var = input_var.cuda()
        batch_graph=batch_graph.cuda()

    # Run words through encoder
    encoder_outputs, problem_output = encoder(input_var, input_length, batch_graph)

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    
    return encoder_outputs, problem_output,all_nums_encoder_outputs

def get_decoder_outputs(encoder_outputs, problem_output,all_nums_encoder_outputs, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
                predict, generate, merge,  output_lang, num_pos, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)
    seq_mask=seq_mask.bool()

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)
    num_mask=num_mask.bool()

    unk = output_lang.word2index["UNK"]


    target = torch.LongTensor(target_batch).transpose(0, 1)
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)



    if USE_CUDA:
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    # max_target_length = max(target_length)
    max_target_length=target.size(0)

    all_node_outputs = []
    # all_leafs = []

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    return all_node_outputs,target

def train_tree_with_knowledge_distill(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos, batch_graph, t_encoder,t_predict,t_generate,t_merge, english=False):


    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()


    encoder_outputs,problem_output,all_node_outputs,target=get_encoder_decoder_outputs(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge,output_lang, num_pos,batch_graph, english=english)

    #teacher
    t_encoder_outputs,t_problem_output,t_all_node_outputs,t_target=get_encoder_decoder_outputs(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               t_encoder, t_predict, t_generate, t_merge,output_lang, num_pos,batch_graph, english=english)
    t_encoder_outputs=t_encoder_outputs.detach()
    t_problem_output=t_problem_output.detach()
    t_all_node_outputs=t_all_node_outputs.detach()
    t_target=t_target.detach()


    #contrastive loss
    # closs=contrastive_loss(problem_output)

    #KD loss
    T=1
    alpha=0.1
    kd_loss= masked_soft_cross_entropy(all_node_outputs/T,t_all_node_outputs/T,target_length)

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)

    #KD loss + loss
    loss=loss+kd_loss*alpha 

    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def train_tree_with_knowledge_distill_use_teacher_label(input_batch, input_length, target_batch, target_length,teacher_batch,distill_weights, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos,batch_graph, t_encoder,t_predict,t_generate,t_merge, english=False,soft_distill=True,hard_distill=True,hidden_distill=False,
               alpha=1.0,beta=0.1,gamma=0.05):


    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()


    encoder_outputs,problem_output,all_node_outputs,target=get_encoder_decoder_outputs(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge,output_lang, num_pos,batch_graph, english=english)

    #teacher
    t_encoder_outputs,t_problem_output,t_all_node_outputs,t_target=get_encoder_decoder_outputs(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               t_encoder, t_predict, t_generate, t_merge,output_lang, num_pos,batch_graph, english=english)
    t_encoder_outputs=t_encoder_outputs.detach()
    t_problem_output=t_problem_output.detach()
    t_all_node_outputs=t_all_node_outputs.detach()
    t_target=t_target.detach()


    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    ce_loss = masked_cross_entropy(all_node_outputs, target, target_length)

    loss=ce_loss*alpha

    #contrastive loss
    # closs=contrastive_loss(problem_output)

    #KD loss
    T=1
    delta=1
    distill_weights=torch.FloatTensor(distill_weights).to(all_node_outputs.device)
    # distill_weights=torch.pow(distill_weights,delta)
    kd_loss= masked_soft_cross_entropy_with_weight(all_node_outputs/T,t_all_node_outputs/T,target_length,distill_weights)

    device=all_node_outputs.device
    if hard_distill:
        #distill from teacher labels
        kd_loss2=[]
        for i,(teacher_labels,teacher_lens) in enumerate(teacher_batch):
            for tea_label,tea_len in zip(teacher_labels,teacher_lens):
                tea_label=torch.LongTensor([tea_label]).to(device)
                tea_len=torch.LongTensor([tea_len])
                if tea_label.size(1)>all_node_outputs.size(1):
                    print(tea_len.item(),max(target_length),all_node_outputs.size(),tea_label.size())
                l=masked_cross_entropy(all_node_outputs[i:i+1],tea_label,tea_len)
                kd_loss2.append(l)
        kd_loss2=torch.stack(kd_loss2).mean()


    #KD loss + loss
    if hard_distill:
        # loss=loss+kd_loss2*0.1
        loss=loss+kd_loss2*beta

    if soft_distill:
        loss=loss+kd_loss*gamma


    if hidden_distill:
        mse=nn.MSELoss(reduction='none')
        mse_losses=mse(problem_output,t_problem_output).mean(dim=1)
        mse_losses=mse_losses*distill_weights
        mse_loss=mse_loss.mean()

        loss=loss+mse_loss*0.02


    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def compute_kd_weight(losses):
    losses=losses.detach()
    median=losses.median()
    weight=torch.where(losses < median, torch.tensor(1).to(losses.device), torch.tensor(0).to(losses.device))
    return weight

def train_tree_with_knowledge_distill_adaptive(input_batch, input_length, target_batch, target_length,teacher_batch,distill_weights, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos,batch_graph, t_encoder,t_predict,t_generate,t_merge, english=False,soft_distill=True,hard_distill=True,hidden_distill=False,
               alpha=1.0,beta=0.1,gamma=0.05):


    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()


    encoder_outputs,problem_output,all_node_outputs,target=get_encoder_decoder_outputs(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge,output_lang, num_pos,batch_graph, english=english)

    #teacher
    t_encoder_outputs,t_problem_output,t_all_node_outputs,t_target=get_encoder_decoder_outputs(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               t_encoder, t_predict, t_generate, t_merge,output_lang, num_pos,batch_graph, english=english)
    t_encoder_outputs=t_encoder_outputs.detach()
    t_problem_output=t_problem_output.detach()
    t_all_node_outputs=t_all_node_outputs.detach()
    t_target=t_target.detach()


    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    ce_loss = masked_cross_entropy(all_node_outputs, target, target_length)

    loss=ce_loss*alpha

    #contrastive loss
    # closs=contrastive_loss(problem_output)

    #KD loss
    if soft_distill:
        distill_weights=torch.FloatTensor(distill_weights).to(all_node_outputs.device)
        soft_kd_loss= masked_soft_cross_entropy_batch(all_node_outputs,t_all_node_outputs,target_length)
        soft_weights=distill_weights*compute_kd_weight(soft_kd_loss)
        

        soft_kd_loss=(soft_kd_loss*soft_weights).sum()/(soft_weights.sum()+1e-12)
        loss=loss+soft_kd_loss*gamma

    device=all_node_outputs.device
    if hard_distill:
        #distill from teacher labels
        hard_kd_loss=[]
        for i,(teacher_labels,teacher_lens) in enumerate(teacher_batch):
            for tea_label,tea_len in zip(teacher_labels,teacher_lens):
                tea_label=torch.LongTensor([tea_label]).to(device)
                tea_len=torch.LongTensor([tea_len])
                if tea_label.size(1)>all_node_outputs.size(1):
                    print(tea_len.item(),max(target_length),all_node_outputs.size(),tea_label.size())
                l=masked_cross_entropy(all_node_outputs[i:i+1],tea_label,tea_len)
                hard_kd_loss.append(l)
        hard_kd_loss=torch.stack(hard_kd_loss)
        #adaptive weight
        hard_weight=compute_kd_weight(hard_kd_loss)

        hard_kd_loss=(hard_kd_loss*hard_weight).sum()/(hard_weight.sum()+1e-12)
        loss=loss+hard_kd_loss*beta

        


    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def compute_cosine_weight(vec1,vec2):
    vec1=f.softmax(vec1,dim=-1)
    vec2=f.softmax(vec2,dim=-1)
    sims=nn.functional.cosine_similarity(vec1,vec2).detach()
    sims=sims.mean(1)
    weight=1-sims
    
    return weight

def train_tree_with_knowledge_distill_adaptive_cosine(input_batch, input_length, target_batch, target_length,teacher_batch,distill_weights, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos,batch_graph, t_encoder,t_predict,t_generate,t_merge, english=False,soft_distill=True,hard_distill=True,hidden_distill=False,
               alpha=1.0,beta=0.1,gamma=0.05):


    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()


    encoder_outputs,problem_output,all_node_outputs,target=get_encoder_decoder_outputs(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge,output_lang, num_pos,batch_graph, english=english)

    #teacher
    t_encoder_outputs,t_problem_output,t_all_node_outputs,t_target=get_encoder_decoder_outputs(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               t_encoder, t_predict, t_generate, t_merge,output_lang, num_pos,batch_graph, english=english)
    t_encoder_outputs=t_encoder_outputs.detach()
    t_problem_output=t_problem_output.detach()
    t_all_node_outputs=t_all_node_outputs.detach()
    t_target=t_target.detach()


    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    ce_loss = masked_cross_entropy(all_node_outputs, target, target_length)

    loss=ce_loss*alpha

    #KD loss
    if soft_distill:
        distill_weights=torch.FloatTensor(distill_weights).to(all_node_outputs.device)
        soft_kd_loss= masked_soft_cross_entropy_batch(all_node_outputs,t_all_node_outputs,target_length)
        soft_weights=distill_weights*compute_cosine_weight(all_node_outputs.detach(),t_all_node_outputs)
        soft_kd_loss=(soft_kd_loss*soft_weights).mean()
        loss=loss+soft_kd_loss*gamma

    device=all_node_outputs.device
    if hard_distill:
        #distill from teacher labels
        hard_kd_loss=[]
        for i,(teacher_labels,teacher_lens) in enumerate(teacher_batch):
            for tea_label,tea_len in zip(teacher_labels,teacher_lens):
                tea_label=torch.LongTensor([tea_label]).to(device)
                tea_len=torch.LongTensor([tea_len])
                if tea_label.size(1)>all_node_outputs.size(1):
                    print(tea_len.item(),max(target_length),all_node_outputs.size(),tea_label.size())
                l=masked_cross_entropy(all_node_outputs[i:i+1],tea_label,tea_len)
                hard_kd_loss.append(l)
        hard_kd_loss=torch.stack(hard_kd_loss)
        #adaptive weight
        # hard_weight=compute_kd_weight(hard_kd_loss)
        # hard_kd_loss=(hard_kd_loss*hard_weight).sum()/(hard_weight.sum()+1e-12)

        hard_kd_loss=hard_kd_loss.mean()

        loss=loss+hard_kd_loss*beta

        


    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()

def kl_divergence(mu1,log_var1,mu2,log_var2):
    # KLD = (log_var_prior - log_var_post + (sigma_post * sigma_post + (mu_post - mu_prior) * (mu_post - mu_prior)) / (
    #             2.0 * sigma_prior * sigma_prior) - 0.5)
    # KLD = torch.mean(KLD, dim=1).mean()
    var1=torch.exp(log_var1)
    var2=torch.exp(log_var2)
    log_var2_var1=log_var2-log_var1
    kld=0.5*(log_var2_var1+(var1+(mu2-mu1)**2)/(var2+1e-12)-1)
    kld=torch.sum(kld,dim=1).mean()
    # torch.distributions.kl_divergence
    # torch.distributions.Normal
    return kld

def train_tree_with_knowledge_distill_logit(input_batch, input_length, target_batch, target_length,teacher_batch,distill_weights, logit_batch,
            nums_stack_batch, num_size_batch, generate_nums,encoder, predict, generate, merge,eq_encoder, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer,eq_optimizer, output_lang, num_pos,batch_graph, english=False,soft_distill=True,hard_distill=True,
               alpha=1.0,beta=0.1,gamma=0.05,kl_weight=1.0):
    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    eq_encoder.train()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    eq_optimizer.zero_grad()

    encoder_outputs,problem_output,mu_prior,log_var_prior,all_node_outputs,target,mu_post,log_var_post=get_encoder_decoder_outputs_vae(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge,eq_encoder,output_lang, num_pos,batch_graph, english=english)

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    ce_loss = masked_cross_entropy(all_node_outputs, target, target_length)

    loss=ce_loss*alpha

    #vae KL-divergence
    # KLD=0.5*torch.sum(torch.exp(log_var)+torch.pow(mu,2)-1.0-log_var)

    sigma_prior = torch.exp(log_var_prior*0.5)
    sigma_post = torch.exp(log_var_post*0.5)

    #KL 1 可用
    # KLD = ( log_var_prior - log_var_post + (sigma_post*sigma_post + (mu_post - mu_prior)*(mu_post - mu_prior)) / (2.0*sigma_prior*sigma_prior) - 0.5)
    # KLD=torch.sum(KLD,dim=1).mean()

    #KL2
    # p=torch.distributions.Normal(mu_post,sigma_post)
    # q=torch.distributions.Normal(mu_prior,sigma_prior)
    # KLD=torch.distributions.kl_divergence(p,q)
    # KLD=torch.mean(KLD)

    #KL 3 可用
    KLD=kl_divergence(mu_post,log_var_post,mu_prior,log_var_prior)

    #KL 4
    # KLD1=0.5*torch.sum(torch.exp(log_var_post)+torch.pow(mu_post,2)-1.0-log_var_post)
    # KLD2=0.5*torch.sum(torch.exp(log_var_prior)+torch.pow(mu_prior,2)-1.0-log_var_prior)
    # KLD1=torch.mean(KLD1)
    # KLD2=torch.mean(KLD2)
    # KLD=KLD1+KLD2

    # KLD=torch.mean(KLD,dim=1).mean()
    loss=loss+KLD*kl_weight
    # loss=loss+KLD*0.1

    #KD loss
    if soft_distill:
        #teacher
        t_all_node_outputs=torch.zeros_like(all_node_outputs).fill_(-1e12)
        for i,logit in enumerate(logit_batch):
            a,b=logit.shape
            t_all_node_outputs[i,:a,:b]=torch.tensor(logit).to(all_node_outputs.device)
        distill_weights=torch.FloatTensor(distill_weights).to(all_node_outputs.device)
        soft_kd_loss= masked_soft_cross_entropy_with_weight(all_node_outputs,t_all_node_outputs,target_length,distill_weights)
        loss=loss+soft_kd_loss*gamma 

    device=all_node_outputs.device
    if hard_distill and len(teacher_batch["input"])>0:
        #distill from teacher labels
        inp=teacher_batch["input"]
        inpLens=teacher_batch["inputLens"]
        tar=teacher_batch["target"]
        tarLens=teacher_batch["targetLens"]
        num_stack=teacher_batch["num_stack"]
        num_size=teacher_batch["num_size"]
        num_pos_t=teacher_batch["num_pos"]
        tea_graph=teacher_batch["graph"]

        _, _, mu_prior_t, log_var_prior_t, all_node_outputs_t, target_t, mu_post_t, log_var_post_t = get_encoder_decoder_outputs_vae(
            inp, inpLens, tar, tarLens, num_stack, num_size, generate_nums,encoder, predict, generate, merge, eq_encoder, output_lang, num_pos_t,tea_graph, english=english)



        hard_kd_loss=masked_cross_entropy(all_node_outputs_t, target_t, tarLens)

        KLD_tea = kl_divergence(mu_post_t, log_var_post_t, mu_prior_t, log_var_prior_t)

        loss=loss+hard_kd_loss*beta+KLD_tea*beta*kl_weight

    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 10)
    torch.nn.utils.clip_grad_norm_(eq_encoder.parameters(),10)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    eq_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def train_tree_with_knowledge_distill_logit_2decoder(input_batch, input_length, target_batch, target_length,teacher_batch,distill_weights1,distill_weights2, logit_batch1,logit_batch2,
            nums_stack_batch, num_size_batch, generate_nums,encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer,merge_optimizer,
            output_lang, num_pos,batch_graph, english=False,soft_distill=True,hard_distill=True,alpha=1.0,beta=0.1,gamma=0.05):
    encoder.train()
    predict.train()
    generate.train()
    merge.train()


    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()


    encoder_outputs,problem_output,all_node_outputs,target=get_encoder_decoder_outputs(input_batch, input_length, target_batch, target_length, \
            nums_stack_batch, num_size_batch, generate_nums,encoder, predict, generate, merge,output_lang, num_pos,batch_graph, english=english)

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    ce_loss = masked_cross_entropy(all_node_outputs, target, target_length)

    loss=ce_loss*alpha

    #KD loss
    if soft_distill:
        #teacher
        t_all_node_outputs1=torch.zeros_like(all_node_outputs).fill_(-1e12)
        t_all_node_outputs2=torch.zeros_like(all_node_outputs).fill_(-1e12)
        for i,logit in enumerate(logit_batch1):
            a,b=logit.shape
            t_all_node_outputs1[i,:a,:b]=torch.tensor(logit).to(all_node_outputs.device)
        for i,logit in enumerate(logit_batch2):
            a,b=logit.shape
            t_all_node_outputs2[i,:a,:b]=torch.tensor(logit).to(all_node_outputs.device)

        distill_weights1=torch.FloatTensor(distill_weights1).to(all_node_outputs.device)
        distill_weights2=torch.FloatTensor(distill_weights2).to(all_node_outputs.device)
        soft_kd_loss1= masked_soft_cross_entropy_with_weight(all_node_outputs,t_all_node_outputs1,target_length,distill_weights1)
        soft_kd_loss2= masked_soft_cross_entropy_with_weight(all_node_outputs,t_all_node_outputs2,target_length,distill_weights2)
        soft_kd_loss=(soft_kd_loss1+soft_kd_loss2)/2
        loss=loss+soft_kd_loss*gamma

    device=all_node_outputs.device
    if hard_distill:
        #distill from teacher labels
        hard_kd_loss=[]
        for i,(teacher_labels,teacher_lens) in enumerate(teacher_batch):
            for tea_label,tea_len in zip(teacher_labels,teacher_lens):
                tea_label=torch.LongTensor([tea_label]).to(device)
                tea_len=torch.LongTensor([tea_len])
                if tea_label.size(1)>all_node_outputs.size(1):
                    print(tea_len.item(),max(target_length),all_node_outputs.size(),tea_label.size())
                l=masked_cross_entropy(all_node_outputs[i:i+1],tea_label,tea_len)
                hard_kd_loss.append(l)
        hard_kd_loss=torch.stack(hard_kd_loss)
        #adaptive weight
        # hard_weight=compute_kd_weight(hard_kd_loss)
        # hard_kd_loss=(hard_kd_loss*hard_weight).sum()/(hard_weight.sum()+1e-12)

        hard_kd_loss=hard_kd_loss.mean()

        loss=loss+hard_kd_loss*beta

    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def train_ro_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, bert, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, bert_optimizer, output_lang, num_pos, batch_graph, mat, char_length, seg_batches, ori_datas, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    mat = torch.FloatTensor(np.array(mat))
    # print("mat size:",mat.size(),"input size:",input_var.size(),"char_length",char_length)

    batch_graph = torch.LongTensor(batch_graph)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    bert.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        mat = mat.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    bert.zero_grad()
    # Run words through encoder

    word_emb, sent_emb = bert(ori_datas, char_length, mat, True)  # B x S x 512

    encoder_outputs, problem_output = encoder(word_emb, input_length, batch_graph)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
    # print(all_node_outputs)

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    bert_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def get_encoder_decoder_outputs_bert_vae(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, bert,eq_encoder, output_lang, num_pos, batch_graph, mat, char_length, seg_batches, ori_datas, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    mat = torch.FloatTensor(np.array(mat))

    batch_graph = torch.LongTensor(batch_graph)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)



    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        mat = mat.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()

    # Run words through encoder
    word_emb, sent_emb = bert(ori_datas, char_length, mat, True)  # B x S x 512

    encoder_outputs, problem_output,z, mu, log_var = encoder(word_emb, input_length, batch_graph)

    #posterior vae
    teacher_target = copy.deepcopy(target)
    eq_encoder_outputs, problem_output_post,z_post,mu_post,log_var_post = eq_encoder(problem_output,teacher_target.cuda(),target_length)

    # 缓解训练和测试阶段不一致问题
    if random.randint(0,10)>4:
        z_post=z
        
    problem_output=z_post

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    # max_target_length = max(target_length)
    max_target_length=target.size(0)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
    # print(all_node_outputs)

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()


    return encoder_outputs, problem_output,mu,log_var,all_node_outputs,target,mu_post,log_var_post



def kl_divergence(mu1,log_var1,mu2,log_var2):
    # KLD = (log_var_prior - log_var_post + (sigma_post * sigma_post + (mu_post - mu_prior) * (mu_post - mu_prior)) / (
    #             2.0 * sigma_prior * sigma_prior) - 0.5)
    # KLD = torch.mean(KLD, dim=1).mean()
    var1=torch.exp(log_var1)
    var2=torch.exp(log_var2)
    log_var2_var1=log_var2-log_var1
    kld=0.5*(log_var2_var1+(var1+(mu2-mu1)**2)/(var2+1e-12)-1)
    kld=torch.sum(kld,dim=1).mean()
    # torch.distributions.kl_divergence
    # torch.distributions.Normal
    return kld

def train_ro_tree_with_KD_VAE(input_batch, input_length, target_batch, target_length, teacher_batch,distill_weights, logit_batch,
                              nums_stack_batch, num_size_batch, generate_nums, encoder, predict, generate, merge, bert,eq_encoder,
                            encoder_optimizer, predict_optimizer, generate_optimizer,merge_optimizer, bert_optimizer, eq_optimizer,
                            output_lang, num_pos, batch_graph, mat, char_length, seg_batches, ori_datas, english=False,soft_distill=True,hard_distill=True,
               alpha=1.0,beta=0.1,gamma=0.05,kl_weight=1.0):
    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    bert.train()
    eq_encoder.train()


    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    bert.zero_grad()
    eq_optimizer.zero_grad()
 
    encoder_outputs,problem_output,mu_prior,log_var_prior,all_node_outputs,target,mu_post,log_var_post=get_encoder_decoder_outputs_bert_vae(
        input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
        encoder, predict, generate, merge,bert,eq_encoder,output_lang, num_pos,batch_graph, mat, char_length, seg_batches, ori_datas,english=english)


    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    # print(all_node_outputs.size(),target.max(),target.size())
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1

    #vae KL-divergence
    # KLD=0.5*torch.sum(torch.exp(log_var)+torch.pow(mu,2)-1.0-log_var)

    sigma_prior = torch.exp(log_var_prior*0.5)
    sigma_post = torch.exp(log_var_post*0.5)

    #KL 1 可用
    # KLD = ( log_var_prior - log_var_post + (sigma_post*sigma_post + (mu_post - mu_prior)*(mu_post - mu_prior)) / (2.0*sigma_prior*sigma_prior) - 0.5)
    # KLD=torch.sum(KLD,dim=1).mean()

    #KL2
    # p=torch.distributions.Normal(mu_post,sigma_post)
    # q=torch.distributions.Normal(mu_prior,sigma_prior)
    # KLD=torch.distributions.kl_divergence(p,q)
    # KLD=torch.mean(KLD)

    #KL 3 可用
    KLD=kl_divergence(mu_post,log_var_post,mu_prior,log_var_prior)

    #KL 4
    # KLD1=0.5*torch.sum(torch.exp(log_var_post)+torch.pow(mu_post,2)-1.0-log_var_post)
    # KLD2=0.5*torch.sum(torch.exp(log_var_prior)+torch.pow(mu_prior,2)-1.0-log_var_prior)
    # KLD1=torch.mean(KLD1)
    # KLD2=torch.mean(KLD2)
    # KLD=KLD1+KLD2

    # KLD=torch.mean(KLD,dim=1).mean()
    loss=loss+KLD*kl_weight
    # loss=loss+KLD*0.1

    #KD loss
    if soft_distill:
        #teacher
        t_all_node_outputs=torch.zeros_like(all_node_outputs).fill_(-1e12)
        for i,logit in enumerate(logit_batch):
            a,b=logit.shape
            t_all_node_outputs[i,:a,:b]=torch.tensor(logit).to(all_node_outputs.device)
        distill_weights=torch.FloatTensor(distill_weights).to(all_node_outputs.device)
        soft_kd_loss= masked_soft_cross_entropy_with_weight(all_node_outputs,t_all_node_outputs,target_length,distill_weights)
        loss=loss+soft_kd_loss*gamma 

    device=all_node_outputs.device
    if hard_distill and len(teacher_batch["input"])>0:
        #distill from teacher labels
        inp=teacher_batch["input"]
        inpLens=teacher_batch["inputLens"]
        tar=teacher_batch["target"]
        tarLens=teacher_batch["targetLens"]
        num_stack=teacher_batch["num_stack"]
        num_size=teacher_batch["num_size"]
        num_pos_t=teacher_batch["num_pos"]
        tea_graph=teacher_batch["graph"]

        mat_t=teacher_batch["mat"]
        seg_t=teacher_batch["seg"]
        char_length_t=teacher_batch["char_length"]
        ori_datas_t=teacher_batch["ori_data"]

        # print(mat_t[0].shape,mat[0].shape,max(char_length_t),max(char_length))

        _, _, mu_prior_t, log_var_prior_t, all_node_outputs_t, target_t, mu_post_t, log_var_post_t = get_encoder_decoder_outputs_bert_vae(
            inp, inpLens, tar, tarLens, num_stack, num_size, generate_nums,encoder, predict, generate, merge,bert, eq_encoder, output_lang, 
            num_pos_t,tea_graph, mat_t, char_length_t, seg_t, ori_datas_t,english=english)

        hard_kd_loss=masked_cross_entropy(all_node_outputs_t, target_t, tarLens)
        KLD_tea = kl_divergence(mu_post_t, log_var_post_t, mu_prior_t, log_var_prior_t)
        loss=loss+hard_kd_loss*beta+KLD_tea*beta*kl_weight


    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    bert_optimizer.step()
    eq_optimizer.step()



    return loss.item()  # , loss_0.item(), loss_1.item()





def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos,batch_graph, english=False):

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()


    encoder_outputs,problem_output,all_node_outputs,target=get_encoder_decoder_outputs(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge,output_lang, num_pos, batch_graph,english=english)

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def train_tree_2decoder(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict1, generate1, merge1,predict2, generate2, merge2, encoder_optimizer, predict_optimizer1, generate_optimizer1,merge_optimizer1, 
               predict_optimizer2, generate_optimizer2,merge_optimizer2, output_lang, num_pos,batch_graph, english=False):

    encoder.train()
    predict1.train()
    generate1.train()
    merge1.train()
    predict2.train()
    generate2.train()
    merge2.train()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer1.zero_grad()
    generate_optimizer1.zero_grad()
    merge_optimizer1.zero_grad()
    predict_optimizer2.zero_grad()
    generate_optimizer2.zero_grad()
    merge_optimizer2.zero_grad()


    encoder_outputs, problem_output,all_node_outputs1,all_node_outputs2,target=get_encoder_decoder_outputs_2decoder(input_batch, input_length, target_batch, target_length, \
            nums_stack_batch, num_size_batch, generate_nums,encoder, predict1, generate1, merge1,predict2, generate2, merge2,output_lang, num_pos,batch_graph, english=english)


    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss1 = masked_cross_entropy(all_node_outputs1, target, target_length)
    loss2 = masked_cross_entropy(all_node_outputs2, target, target_length)

    loss=loss1+loss2
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer1.step()
    generate_optimizer1.step()
    merge_optimizer1.step()
    predict_optimizer2.step()
    generate_optimizer2.step()
    merge_optimizer2.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,batch_graph,
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    seq_mask=seq_mask.bool()
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)
    batch_graph = torch.LongTensor(batch_graph)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)
    num_mask=num_mask.bool()

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph=batch_graph.cuda()
    # Run words through encoder

    encoder_outputs, problem_output,z,mu,log_var= encoder(input_var, [input_length], batch_graph)

    # problem_output=problem_output+z

    # Prepare input and output variables
    node_stacks = [[TreeNode(root)] for root in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out


def evaluate_ro_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge,bert,
                  output_lang, num_pos, batch_graph, mat, ori_datas, beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH,returen_beam=False):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)
    batch_graph = torch.LongTensor(batch_graph)

    mat = torch.FloatTensor(np.array([mat]))

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    bert.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()
        mat = mat.cuda()
    # Run words through encoder

    # Run words through encoder
    word_emb, sent_emb = bert(' '.join(ori_datas), [mat.shape[-1]], mat,
                              out_all_hidden=True)  # B[1] * char_len_max * 512

    encoder_outputs, problem_output = encoder(word_emb, [input_length], batch_graph)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break
    if returen_beam:
        return beams
    return beams[0].out

def evaluate_ro_tree_vae(input_batch, input_length, generate_nums, encoder, predict, generate, merge,bert,
                  output_lang, num_pos, batch_graph, mat, ori_datas, beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH,returen_beam=False):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)
    batch_graph = torch.LongTensor(batch_graph)

    mat = torch.FloatTensor(np.array([mat]))

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    bert.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()
        mat = mat.cuda()
    # Run words through encoder

    # Run words through encoder
    word_emb, sent_emb = bert(' '.join(ori_datas), [mat.shape[-1]], mat,
                              out_all_hidden=True)  # B[1] * char_len_max * 512

    encoder_outputs, problem_output,z,mu,log_var = encoder(word_emb, [input_length], batch_graph)
    problem_output=z

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break
    if returen_beam:
        return beams
    return beams[0].out


def evaluate_tree_beam(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,batch_graph,
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    seq_mask=seq_mask.bool()
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)
    batch_graph = torch.LongTensor(batch_graph)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)
    num_mask=num_mask.bool()

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph=batch_graph.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length], batch_graph)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams

def topdown_train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch,
                       generate_nums, encoder, predict, generate, encoder_optimizer, predict_optimizer,
                       generate_optimizer, output_lang, num_pos, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        for idx, l, r, node_stack, i in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                            node_stacks, target[t].tolist()):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def topdown_evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, output_lang, num_pos,
                          beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, embeddings_stacks, left_childs,
                                              current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out
