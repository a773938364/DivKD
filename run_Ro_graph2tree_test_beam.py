# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import os
import torch.optim
from src.expressions_transfer import *
import json
import random
import tqdm
import numpy as np
import pickle
# from ltp import LTP

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import argparse

parser = argparse.ArgumentParser(description='Example of using argparse to configure hyperparameters')
parser.add_argument('--seed', type=int, default=19941225, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epoch', type=int, default=80, help='Number of training epochs')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer')
parser.add_argument('--alpha', type=float, default=1.0, help='Weight for CE loss')
parser.add_argument('--beta', type=float, default=0.1, help='Weight for Hard KD loss')
parser.add_argument('--gamma', type=float, default=0.05, help='Weight for Soft KD loss')
parser.add_argument('--use_soft_kd',action='store_true',help="Soft distill")
parser.add_argument('--use_hard_kd',action='store_true',help="hard distill")
parser.add_argument('--teacher_model_path',type=str,default="models/model_teacher_0.7439")
parser.add_argument('--student_model_path',type=str,default="models/model_student")
parser.add_argument('--Roberta_path', type=str,default=r"/home/ubuntu/projects/PLMs/roberta-base-en")
# parser.add_argument('--Roberta_path', type=str,default=r"H:/PreTrainedLM/LM/roberta-base-en")
parser.add_argument('--ltp_path', type=str,default=r"/home/zy/research_v2/tool/Ltp_base2_v3_")

args = parser.parse_args()

# Set the seed for reproducibility
set_seed(args.seed)

batch_size = args.batch_size
print('随机种子：', args.seed)
print('-'*100)
embedding_size = 128
hidden_size = 512
bert_hidden = 768
bert_learning_rate = 5e-5
bert_weight_decay = 5e-5
n_epochs = args.epoch
learning_rate = args.lr
weight_decay = args.weight_decay
beam_size = 5
n_layers = 2
get_trans_flag = True
ori_path = './data/'
prefix = '_PROCESS.json'
# ltp = LTP(path=args.ltp_path)

def transfer_ro_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = []
    generate_nums = []
    generate_nums_dict={}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        id=d["id"]
        seg = d["corenlp_seg_wo_plus"]#.strip().split(" ")
        equations = d["equation"]

        eq_segs = d["num_exp"]

        for s in eq_segs:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1


        parse = d["parse"]

        num_seg = d["num_seg"]
        num_pos = d["num_pos"]
        if num_pos is None:
            print(num_pos)
        nums=d["nums"]
        if nums is None:
            print(nums)
        # assert nums is not None,d
        if copy_nums < len(nums):
            copy_nums = len(nums)

        # num_pos = []
        # for i, j in enumerate(input_seq):
        #     if j == "NUM":
        #         num_pos.append(i)

        if len(nums) != 0:
            # pairs.append((input_seq, eq_segs, nums, num_pos))
            item={
                "id":id,
                "original_text": seg,
                "num_text": num_seg,
                'infix_equation': eq_segs,
                "parse":parse,
                "nums":nums,
                'num_pos': num_pos,
                "ans":d["answer"]
            }
            pairs.append(item)

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 4:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def get_ro_train_test_fold(ori_path,prefix,pairs,group):
    mode_train = 'train'
    mode_valid = 'dev'
    mode_test = 'test'
    train_path = ori_path + "NEAT_%s_PROCESS.json"%mode_train # data/train23k_processed.json
    valid_path = ori_path + "NEAT_%s_PROCESS.json"%mode_valid
    test_path = ori_path + "NEAT_%s_PROCESS.json"%mode_test
    train = read_json(train_path)
    train_id = ["train_%s"%item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = ["dev_%s"%item['id'] for item in valid]
    test = read_json(test_path)
    test_id = ["test_%s"%item['id'] for item in test]


    train_fold = []
    valid_fold = []
    test_fold = []
    for p in pairs:
        p = list(p)
        idx = p[-1]
        for g in group:
            # print(idx)
            if g['id'] == idx:
                # p[-1]=int(idx.split("_")[-1])
                # p.append(g['id'].split("_")[-1])
                p.append(g['group_num'])
                p = tuple(p)
                if idx in train_id:
                    train_fold.append(p)
                elif idx in test_id:
                    test_fold.append(p)
                else:
                    # print(idx)
                    valid_fold.append(p)
    return train_fold, test_fold, valid_fold


def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num

def load_mathqa_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)

    return data
data = load_mathqa_data("data/NEAT_combine.json")
group_data = read_json("data/NEAT_combine_PROCESS.json")

pairs, generate_nums, copy_nums = transfer_ro_num(data)

temp_pairs = []
for p in pairs:
    if len(p['num_text']) != len(p['original_text']):
        assert 0==1
    temp_pairs.append((p['num_text'], from_infix_to_prefix(p['infix_equation']), p['nums'], p['num_pos'], p['parse'],
                       p['original_text'], p['id']))
pairs = temp_pairs
# print(pairs[1])
# print(group_data[1])

# pairs = sorted(pairs, key=lambda tp:tp[-1], reverse=False)

# print(pairs[1])
train_fold, test_fold, valid_fold = get_ro_train_test_fold(ori_path, prefix, pairs, group_data)
print(len(train_fold), len(test_fold), len(valid_fold))
# print(train_fold[1], test_fold[1], valid_fold[1])

bert = EncoderChar(bert_path=args.Roberta_path, bert_size=bert_hidden, hidden_size=hidden_size, get_word_and_sent=get_trans_flag)
start = time.time()
input_lang, output_lang, train_pairs, test_pairs = prepare_ro_data_en(train_fold, test_fold, 2, generate_nums,
                                                                   copy_nums, bert.tokenizer, tree=True)
print("output_vocab_size:",len(output_lang.index2word))
print('构建input_vocab 与nv output_vocab 和 train and test 数据集  用时: ' + time_since(time.time()-start))
print(output_lang.index2word, output_lang.num_start)

encoder = EncoderSeq_ro(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
bert_optimizer = torch.optim.AdamW(bert.parameters(), lr=bert_learning_rate, weight_decay=bert_weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
bert_scheduler = torch.optim.lr_scheduler.StepLR(bert_optimizer, step_size=20, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()
    bert.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

def load_model(path="models/model_traintest_0.747"):

    encoder_params=torch.load(os.path.join(path,"encoder"))
    predict_params=torch.load(os.path.join(path,"predict"))
    generate_params=torch.load(os.path.join(path,"generate"))
    merge_params=torch.load(os.path.join(path,"merge"))
    bert_params=torch.load(os.path.join(path,"bert"))

    encoder.load_state_dict(encoder_params)
    predict.load_state_dict(predict_params)
    generate.load_state_dict(generate_params)
    merge.load_state_dict(merge_params)
    bert.load_state_dict(bert_params)


def evaluate(test_pairs):
    start=time.time()
    value_ac = 0
    value_ac_beam=0
    equation_ac = 0
    eval_total = 0
    beam_results=[]
    all_logits=[]
    for test_batch in test_pairs:
        batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])

        input_batch=[test_batch[0]]
        input_length=[test_batch[1]]
        target_batch=[test_batch[2]]
        target_length=[test_batch[3]]
        num_batch=[test_batch[4]]
        num_pos_batch=[test_batch[5]]
        nums_stack_batch=[test_batch[6]]
        num_size_batch=[len(test_batch[5])]
        
        mat=test_batch[9]
        ori_datas=test_batch[8]

        encoder_outputs,problem_output,all_node_outputs,target=get_encoder_decoder_outputs_evaluate(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, 
            generate_num_ids, bert,encoder, predict, generate, merge,output_lang,  num_pos_batch, batch_graph,mat, ori_datas)
        
        all_node_outputs=all_node_outputs.detach().cpu().numpy()[0]
        # print(all_node_outputs.shape)
        all_logits.append(all_node_outputs)



        test_res_beams = evaluate_ro_tree(test_batch[0], test_batch[1], generate_num_ids,  encoder, predict, generate,
                                    merge,bert, output_lang, test_batch[5],batch_graph, mat,ori_datas,beam_size=beam_size,returen_beam=True)
        beam_res=[]
        beam_res_exp=[]
        all_res_exp=[]
        beam_res_score=[]
        for rank,beam in enumerate(test_res_beams):
            test_res=beam.out
            score=np.exp(beam.score).item()
            val_ac, equ_ac, pred_exp, tar_exp = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            all_res_exp.append([pred_exp,val_ac,score])
            if val_ac:
                value_ac_beam += 1
                if rank==0:
                    value_ac+=1
                beam_res.append(test_res)
                beam_res_score.append((test_res,rank,score))
                beam_res_exp.append(pred_exp)
            if equ_ac:
                equation_ac += 1
        eval_total += 1
        beam_results.append({"label":test_batch[2],"predict":beam_res,"predict_score":beam_res_score,
                             "label_exp":tar_exp,"predict_exp":beam_res_exp,"all_exp":all_res_exp})
    
        
    print(equation_ac, value_ac, eval_total)
    print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
    print("testing time", time_since(time.time() - start))
    print("------------------------------------------------------")
    return beam_results,all_logits

if __name__=="__main__":
    
    model_path="models/model_teacher_0.7439"
    load_model(path=model_path)

    # beam_results,all_logits=evaluate(test_pairs)

    beam_results,all_logits=evaluate(train_pairs)
    with open(os.path.join(model_path,"beam_res_train.json"),"w",encoding="utf-8") as fout:
        json.dump(beam_results,fout,indent=3)
    with open(os.path.join(model_path,"all_logits_train.pkl"),"wb") as fout:
        pickle.dump(all_logits,fout)