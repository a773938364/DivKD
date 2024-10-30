import random
import json
import copy
import re
import numpy as np
from copy import  deepcopy

PAD_token = 0
PAD, CLS = '[PAD]', '[CLS]'


class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data


# remove the superfluous brackets
def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y


def load_mawps_data(filename):  # load the json data to list(dict()) for MAWPS
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data = []
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[2:]
                out_data.append(temp)
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[:-2]
                out_data.append(temp)
                continue
    return out_data


def load_roth_data(filename):  # load the json data to dict(dict()) for roth data
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data = {}
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = remove_brackets(xt)
                    y = temp["sQuestion"]
                    seg = y.strip().split(" ")
                    temp_y = ""
                    for s in seg:
                        if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                            temp_y += s[:-1] + " " + s[-1:] + " "
                        else:
                            temp_y += s + " "
                    temp["sQuestion"] = temp_y[:-1]
                    out_data[temp["iIndex"]] = temp
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = remove_brackets(xt)
                    y = temp["sQuestion"]
                    seg = y.strip().split(" ")
                    temp_y = ""
                    for s in seg:
                        if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                            temp_y += s[:-1] + " " + s[-1:] + " "
                        else:
                            temp_y += s + " "
                    temp["sQuestion"] = temp_y[:-1]
                    out_data[temp["iIndex"]] = temp
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = remove_brackets(x[2:])
                y = temp["sQuestion"]
                seg = y.strip().split(" ")
                temp_y = ""
                for s in seg:
                    if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                        temp_y += s[:-1] + " " + s[-1:] + " "
                    else:
                        temp_y += s + " "
                temp["sQuestion"] = temp_y[:-1]
                out_data[temp["iIndex"]] = temp
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = remove_brackets(x[2:])
                y = temp["sQuestion"]
                seg = y.strip().split(" ")
                temp_y = ""
                for s in seg:
                    if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                        temp_y += s[:-1] + " " + s[-1:] + " "
                    else:
                        temp_y += s + " "
                temp["sQuestion"] = temp_y[:-1]
                out_data[temp["iIndex"]] = temp
                continue
    return out_data

# for testing equation
# def out_equation(test, num_list):
#     test_str = ""
#     for c in test:
#         if c[0] == "N":
#             x = num_list[int(c[1:])]
#             if x[-1] == "%":
#                 test_str += "(" + x[:-1] + "/100.0" + ")"
#             else:
#                 test_str += x
#         elif c == "^":
#             test_str += "**"
#         elif c == "[":
#             test_str += "("
#         elif c == "]":
#             test_str += ")"
#         else:
#             test_str += c
#     return test_str


def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


def transfer_english_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = []
    generate_nums = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # def seg_and_tag(st):  # seg the equation and tag the num
        #     res = []
        #     pos_st = re.search(pattern, st)
        #     if pos_st:
        #         p_start = pos_st.start()
        #         p_end = pos_st.end()
        #         if p_start > 0:
        #             res += seg_and_tag(st[:p_start])
        #         st_num = st[p_start:p_end]
        #         if st_num[-2:] == ".0":
        #             st_num = st_num[:-2]
        #         if "." in st_num and st_num[-1] == "0":
        #             st_num = st_num[:-1]
        #         if nums.count(st_num) == 1:
        #             res.append("N"+str(nums.index(st_num)))
        #         else:
        #             res.append(st_num)
        #         if p_end < len(st):
        #             res += seg_and_tag(st[p_end:])
        #     else:
        #         for sst in st:
        #             res.append(sst)
        #     return res
        # out_seq = seg_and_tag(equations)

        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs.append((input_seq, eq_segs, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums



def gen_word_char_mat(ori_seg, tokenizer):
    # 中文版本，够用了
    length = 1
    word_list = []
    chars = []
    for i, w in enumerate(ori_seg):
        w_char = tokenizer.tokenize(w)
        chars.extend(w_char)
        word_list.append((i, w, w_char, length, length + len(w_char)))
        length += len(w_char)
    #print(word_list)
    max_pos = 0
    pos_s = np.zeros((len(ori_seg),), dtype=np.int64)
    # [0, .., .., ], 单个 最多 latt 的句子
    pos_e = np.zeros((len(ori_seg),), dtype=np.int64)
    mat = np.zeros((len(word_list), length + 1), dtype=np.int64) # len(input) * length_char + 2
    forward_position = np.zeros(len(ori_seg))
    # [0.0, .., .., 0.0], 单个 最多 token 的句子
    backward_position = np.zeros(len(ori_seg))
    # print("gen_word_char_mat:",mat.shape, len(word_list), length)
    # if len(word_list)>length+1:
    #     print("word>char:",ori_seg)
    for i, index in enumerate(word_list):
        s = index[-2]
        e = index[-1]
        pos_s[i] = s
        pos_e[i] = e
        forward_position[i] = s
        backward_position[i] = e
        max_pos = e if e > max_pos else max_pos
        for j in range(s, e):
            mat[i][j] = 1
    return mat, forward_position, backward_position, pos_s, pos_e, chars


def get_num_char_mat(num_pos, word_char_mat):
    mat = copy.deepcopy(word_char_mat)
    num_char_mat = []
    for n in num_pos:
        char_list = mat[n]
        temp_index = []
        for i, c in enumerate(char_list):
            if c == 1:
                temp_index.append(i)
        num_char_mat.append((n, temp_index))
    return num_char_mat


def prepare_ro_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, cutting, tree=False):
    '''
    (['一', '个', '工程队', '挖土', '，', '第一天', '挖', '了', 'NUM', '方', '，', '从', '第', '二', '天', '开始', '每天', '都', '挖', 'NUM', '方', '，', '连续', '挖', '了', 'NUM', '天', '，', '这个', '工程队', '一周', '共', '挖土', '多少', '方', '？'],
     ['+', 'N0', '*', 'N1', '-', 'N2', '1'],
     ['316', '230', '6'],
     [8, 19, 25],
     [1, 2, 3, -1, 3, 6, 3, 6, 9, 6, 3, 18, 13, 14, 11, 11, 18, 18, 3, 20, 18, 3, 23, 3, 23, 26, 23, 3, 29, 32, 32, 32, 3, 34, 32, 3],
     && ['一', '个', '工程队', '挖土', '，', '第一天', '挖', '了', '316', '方', '，', '从', '第', '二', '天', '开始', '每天', '都', '挖', '230', '方', '，', '连续', '挖', '了', '6', '天', '，', '这个', '工程队', '一周', '共', '挖土', '多少', '方', '？'],
     '2',
     [6, 7, 8, 15, 16, 17, 21, 22, 23],
     '2')
    原本:
    (['一', '项', '工程', '，', '甲', '乙', '两队', '合', '做', 'NUM', '天', '完成', '，', '现在', '甲队', '单独', '做', 'NUM', '天', '后', '，', '乙', '队', '加入', '，', '两队', '又', '合', '做', '了', 'NUM', '天', '，', '这时', '甲队', '调', '走', '，', '乙', '队', '继续', '做', 'NUM', '天才', '完成', '这项', '工程', '．', '甲队', '单独', '做', '这项', '工程', '需', '多少', '天', '？'],
    ['1', '/', '[', '(', '1', '-', '(', '(', 'N2', '+', 'N3', ')', '/', '(', 'N0', ')', ')', ')', '/', '(', 'N1', '-', 'N3', ')', ']'],
    ['30', '24', '12', '15'],
    [9, 17, 30, 42],
    [1, 2, 8, 2, 6, 4, 8, 8, -1, 10, 8, 8, 8, 16, 16, 16, 19, 18, 16, 23, 19, 22, 23, 8, 23, 28, 27, 28, 23, 28, 31, 28, 23, 35, 35, 23, 35, 35, 39, 40, 35, 35, 43, 41, 35, 46, 44, 23, 50, 50, 53, 52, 50, 23, 55, 53, 8],
    && [8, 9, 10, 16, 17, 18, 29, 30, 31, 41, 42, 43, 53, 54, 55],
    ['一', '项', '工程', '，', '甲', '乙', '两队', '合', '做', '30', '天', '完成', '，', '现在', '甲队', '单独', '做', '24', '天', '后', '，', '乙', '队', '加入', '，', '两队', '又', '合', '做', '了', '12', '天', '，', '这时', '甲队', '调', '走', '，', '乙', '队', '继续', '做', '15', '天才', '完成', '这项', '工程', '．', '甲队', '单独', '做', '这项', '工程', '需', '多少', '天', '？'])
    '''
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    print('-'*40, " Indexing words... ", '-'*40)
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    '2022- 3- 13 加入了 output_lang'
    input_lang.build_input_lang(trim_min_count, output_lang.index2word)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)
            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])
        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        mat, _, _, _, _, _ = gen_word_char_mat(pair[5], tokenizer=cutting)
        # [], len(num_pos) 个 (pos, [char_index, char_index,.. ])
        num_dict = get_num_char_mat(pair[3], mat)

        assert mat.shape[0] == len(input_cell)
        train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack, pair[7], pair[5], mat))

    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)
            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])
        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        mat, _, _, _, _, _ = gen_word_char_mat(pair[5], tokenizer=cutting)
        num_dict = get_num_char_mat(pair[3], mat)

        assert mat.shape[0] == len(input_cell)
        # (input_cell, len(input_cell), output_cell, len(output_cell),
        #  pair[2], pair[3], num_stack,pair[4]))
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3], num_stack, pair[7], pair[5], mat))

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    print('output_lang.index2word: ', output_lang.index2word)
    print('Number of testind data %d' % (len(test_pairs)))

    return input_lang, output_lang, train_pairs, test_pairs


def transfer_roth_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = {}
    generate_nums = {}
    copy_nums = 0
    for key in data:
        d = data[key]
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # def seg_and_tag(st):  # seg the equation and tag the num
        #     res = []
        #     pos_st = re.search(pattern, st)
        #     if pos_st:
        #         p_start = pos_st.start()
        #         p_end = pos_st.end()
        #         if p_start > 0:
        #             res += seg_and_tag(st[:p_start])
        #         st_num = st[p_start:p_end]
        #         if st_num[-2:] == ".0":
        #             st_num = st_num[:-2]
        #         if "." in st_num and st_num[-1] == "0":
        #             st_num = st_num[:-1]
        #         if nums.count(st_num) == 1:
        #             res.append("N"+str(nums.index(st_num)))
        #         else:
        #             res.append(st_num)
        #         if p_end < len(st):
        #             res += seg_and_tag(st[p_end:])
        #     else:
        #         for sst in st:
        #             res.append(sst)
        #     return res
        # out_seq = seg_and_tag(equations)

        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs[key] = (input_seq, eq_segs, nums, num_pos)

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res

def prepare_ro_data_en(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, cutting, tree=False):
    '''
    0 (['There', 'are', 'NUM', 'erasers', 'and', 'NUM', 'scissors', 'in', 'the', 'drawer', '.', 'Jason', 'placed', 'NUM', 'erasers', 'in', 'the', 'drawer', '.', 'How', 'many', 'erasers', 'are', 'now', 'there', 'in', 'total', '?'],
    1 ['+', 'N0', 'N2'],
    2 ['139', '118', '131'],
    3 [2, 5, 13],
    4 [[0, 1, 'expl'], [1, -1, 'root'], [2, 3, 'nummod'], [3, 1, 'nsubj'], [4, 6, 'cc'], [5, 6, 'nummod'], [6, 3, 'conj'], [7, 9, 'case'], [8, 9, 'det'], [9, 1, 'obl'], [10, 1, 'punct'], [11, 1, 'nsubj'], [12, -1, 'root'], [13, 3, 'nummod'], [14, 1, 'obj'], [15, 6, 'case'], [16, 6, 'det'], [17, 1, 'obl'], [18, 1, 'punct'], [8, 1, 'advmod'], [9, 2, 'amod'], [10, 5, 'nsubj'], [11, 5, 'cop'], [12, 5, 'advmod'], [13, -1, 'root'], [14, 7, 'case'], [15, 5, 'obl'], [16, 5, 'punct']],
    5 ['There', 'are', '139', 'erasers', 'and', '118', 'scissors', 'in', 'the', 'drawer', '.', 'Jason', 'placed', '131', 'erasers', 'in', 'the', 'drawer', '.', 'How', 'many', 'erasers', 'are', 'now', 'there', 'in', 'total', '?'],
    6 id 386,
    7 group [11, 12, 13, 15, 16, 17, 24, 25, 26, 31, 32, 33])
    '''
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            # num_text
            input_lang.add_sen_to_vocab(pair[0])
            # eq_text
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])

    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    print("output_lang:",output_lang.index2word)
    for pair in pairs_trained:
        parse = [p[1] for p in pair[4]]
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                # nums
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
                print("OOV word:",word,"num_stack:",pair[1],pair[2],num_stack)
            if not flag_not and len(temp_num) == 0:
                print("OOV & not in nums word:", word, "num_stack:", pair[1], pair[2], num_stack)
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        mat, _, _, _, _, _ = gen_english_word_char_mat(pair[5], tokenizer=cutting)
        # print("mat:",mat.shape,len(input_cell))
        assert mat.shape[0] == len(input_cell)
        train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack, pair[7], pair[5], mat))

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        parse = [p[1] for p in pair[4]]
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        mat, _, _, _, _, _ = gen_english_word_char_mat(pair[5], tokenizer=cutting)
        assert mat.shape[0] == len(input_cell)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack, pair[7], pair[5], mat))
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack, pair[4]))
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3], num_stack,pair[4]))
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


def prepare_de_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        input_lang.add_sen_to_vocab(pair[0])
        output_lang.add_sen_to_vocab(pair[1])

    input_lang.build_input_lang(trim_min_count)

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        # train_pairs.append([input_cell, len(input_cell), pair[1], 0, pair[2], pair[3], num_stack, pair[4]])
        train_pairs.append([input_cell, len(input_cell), pair[1], 0, pair[2], pair[3], num_stack])
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3], num_stack))
    print('Number of testind data %d' % (len(test_pairs)))
    # the following is to test out_equation
    # counter = 0
    # for pdx, p in enumerate(train_pairs):
    #     temp_out = allocation(p[2], 0.8)
    #     x = out_equation(p[2], p[4])
    #     y = out_equation(temp_out, p[4])
    #     if x != y:
    #         counter += 1
    #     ans = p[7]
    #     if ans[-1] == '%':
    #         ans = ans[:-1] + "/100"
    #     if "(" in ans:
    #         for idx, i in enumerate(ans):
    #             if i != "(":
    #                 continue
    #             else:
    #                 break
    #         ans = ans[:idx] + "+" + ans[idx:]
    #     try:
    #         if abs(eval(y + "-(" + x + ")")) < 1e-4:
    #             z = 1
    #         else:
    #             print(pdx, x, p[2], y, temp_out, eval(x), eval("(" + ans + ")"))
    #     except:
    #         print(pdx, x, p[2], y, temp_out, p[7])
    # print(counter)
    return input_lang, output_lang, train_pairs, test_pairs


# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq

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

# num net graph
def get_lower_num_graph(max_len, sentence_length, num_list, id_num_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    num_list = change_num(num_list)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    if not contain_zh_flag:
        return graph
    for i in range(len(id_num_list)):
        for j in range(len(id_num_list)):
            if float(num_list[i]) <= float(num_list[j]):
                graph[id_num_list[i]][id_num_list[j]] = 1
            else:
                graph[id_num_list[j]][id_num_list[i]] = 1
    return graph

def get_greater_num_graph(max_len, sentence_length, num_list, id_num_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    num_list = change_num(num_list)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    if not contain_zh_flag:
        return graph
    for i in range(len(id_num_list)):
        for j in range(len(id_num_list)):
            if float(num_list[i]) > float(num_list[j]):
                graph[id_num_list[i]][id_num_list[j]] = 1
            else:
                graph[id_num_list[j]][id_num_list[i]] = 1
    return graph

# attribute between graph
def get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    #quantity_cell_list = quantity_cell_list.extend(id_num_list)
    if not contain_zh_flag:
        return graph
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1
    for i in quantity_cell_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len:
                if input_batch[i] == input_batch[j]:
                    graph[i][j] = 1
                    graph[j][i] = 1
    return graph

# quantity between graph
def get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    #quantity_cell_list = quantity_cell_list.extend(id_num_list)
    if not contain_zh_flag:
        return graph
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1
    for i in id_num_list:
        for j in id_num_list:
            graph[i][j] = 1
            graph[j][i] = 1
    return graph

# quantity cell graph
def get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list,contain_zh_flag=True):
    ## To Check
    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    #quantity_cell_list = quantity_cell_list.extend(id_num_list)
    if not contain_zh_flag:
        return graph
    for i in id_num_list:
        for j in quantity_cell_list:
            # print(i,j)
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1
    return graph

def get_single_batch_graph(input_batch, input_length,group,num_value,num_pos):
    batch_graph = []
    max_len = max(input_length)
    for i in range(len(input_length)):
        input_batch_t = input_batch[i]
        sentence_length = input_length[i]
        quantity_cell_list = group[i]
        num_list = num_value[i]
        id_num_list = num_pos[i]
        graph_newc = get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_greater = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
        graph_lower = get_lower_num_graph(max_len, sentence_length, num_list, id_num_list)
        graph_quanbet = get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_attbet = get_attribute_between_graph(input_batch_t, max_len, id_num_list, sentence_length, quantity_cell_list)
        #graph_newc1 = get_quantity_graph1(input_batch_t, max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_total = [graph_newc.tolist(),graph_greater.tolist(),graph_lower.tolist(),graph_quanbet.tolist(),graph_attbet.tolist()]
        batch_graph.append(graph_total)
    batch_graph = np.array(batch_graph)
    return batch_graph

def get_single_example_graph(input_batch, input_length,group,num_value,num_pos):
    batch_graph = []
    max_len = input_length
    sentence_length = input_length
    quantity_cell_list = group
    num_list = num_value
    id_num_list = num_pos
    graph_newc = get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_quanbet = get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_attbet = get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_greater = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
    graph_lower = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
    #graph_newc1 = get_quantity_graph1(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_total = [graph_newc.tolist(),graph_greater.tolist(),graph_lower.tolist(),graph_quanbet.tolist(),graph_attbet.tolist()]
    batch_graph.append(graph_total)
    batch_graph = np.array(batch_graph)
    return batch_graph

# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    graph_batches = []
    num_value_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _,_ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        group_batch = []
        num_value_batch = []
        for i, li, j, lj, num, num_pos, num_stack, group in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            num_value_batch.append(num)
            group_batch.append(group)
            
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        group_batches.append(group_batch)
        graph_batches.append(get_single_batch_graph(input_batch, input_length,group_batch,num_value_batch,num_pos_batch))
        
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches


def pad_word_char_mat(ori_mat, input_len, char_len):
    unified_mat = np.zeros((input_len, char_len), dtype=np.int64)

    shape_raw = ori_mat.shape

    unified_mat[:shape_raw[0], :shape_raw[1]] = ori_mat

    return unified_mat



# 新版本英文
def gen_english_word_char_mat(seg, tokenizer, robert=True):
    words = deepcopy(seg)
    length = 1
    word_list = []
    chars = []

    # RoBERTa 分词 + 获取偏移量
    encoded = tokenizer(' '.join(words), return_offsets_mapping=True)
    subword_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    tokenLen=len(subword_tokens)

    # 对齐每个单词对应的子词列表
    def align_word_to_subwords(words, subword_tokens, offsets):
        text = " ".join(words)
        pos2tokens = [("", "") for i in range(len(text))]
        for tid, (token, (s, e)) in enumerate(zip(subword_tokens, offsets)):
            for i in range(s, e):
                pos2tokens[i] = (tid, token)
        pos2words = ["" for i in range(len(text))]
        index = 0
        for wid, word in enumerate(words):
            for i in range(index, index + len(word)):
                pos2words[i] = (wid, word)
            if i < len(text) - 1:
                pos2words[i + 1] = ("_", " ")
                index = i + 1 + 1
        # print(pos2tokens)
        # print(pos2words)

        word2tokens = []
        wordId = 0
        wordTokens = []
        for (wid, word), (tid, token) in zip(pos2words, pos2tokens):
            if wid == wordId:
                if tid not in wordTokens:
                    wordTokens.append(tid)
            else:
                word2tokens.append(wordTokens)
                wordId += 1
                wordTokens = []
        word2tokens.append(wordTokens)
        # print(word2tokens)
        word2tokens_new = []
        for wid, word in enumerate(words):
            wtokenIds = word2tokens[wid]
            wtokens = []
            for tid in wtokenIds:
                if tid != "":
                    wtoken = subword_tokens[tid]
                    wtokens.append(wtoken)
            word2tokens_new.append(wtokens)
        # print(word2tokens_new)
        return word2tokens_new
    word_chars=align_word_to_subwords(words,subword_tokens,offsets)
    length=1
    word_list=[]

    for i,(word,subwords) in enumerate(zip(words,word_chars)):
        word_list.append((i,word,subwords,length,length+len(subwords)))
        length+=len(subwords)


    # if length+1!=tokenLen:
    #     print(word_chars)
    #     print("len:",length+1,tokenLen)
    #print(word_list)
    #print('chars : ', chars, len(chars))
    max_pos = 0
    pos_s = np.zeros((len(words),), dtype=np.int32)
    # [0, .., .., ], 单个 最多 latt 的句子
    pos_e = np.zeros((len(words),), dtype=np.int32)
    mat = np.zeros((len(word_list), tokenLen), dtype=np.int32) # len(input) * length_char + 2
    forward_position = np.zeros(len(words))
    # [0.0, .., .., 0.0], 单个 最多 token 的句子
    backward_position = np.zeros(len(words))
    # print("gen_word_char_mat:",mat.shape, len(word_list), length)
    if len(word_list)>length+1:
        print("word%s>char%s: %s"%(len(word_list),length+1,seg))
    for i, index in enumerate(word_list):
        s = index[-2]
        e = index[-1]
        pos_s[i] = s
        pos_e[i] = e
        forward_position[i] = s
        backward_position[i] = e
        max_pos = e if e > max_pos else max_pos
        for j in range(s, e):
            mat[i][j] = 1
    return mat, forward_position, backward_position, pos_s, pos_e, word_chars


def get_word_num_mat(input_len_max, num_pos_batch, max_num_len):
    batch_word2num = []

    for i in range(len(num_pos_batch)):
        graph = np.zeros((input_len_max, max_num_len))

        for j, index in enumerate(num_pos_batch[i]):
            graph[index][j] = 1

        batch_word2num.append(graph)

    return batch_word2num

# prepare the batches
def prepare_ro_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    graph_batches = []
    num_value_batches = []
    mat_batches = []
    seg_batches = []
    ori_batches = []
    char_lengths = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos + batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        char_length = []
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        # pair[2], pair[3], num_stack, pair[7], pair[5], mat, num_dict))
        for _, i, _, j, _, _, _, _, ori_data, mat in batch:
            input_length.append(i)
            output_length.append(j)
            char_length.append(mat.shape[-1])
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        char_len_max = max(char_length)
        # print("maxlen:%s,max char len:%s"%(input_len_max,char_len_max))
        char_lengths.append(char_length)

        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        group_batch = []
        num_value_batch = []
        seg_batch = []
        mat_batch = []
        ori_batch = []

        for i, li, j, lj, num, num_pos, num_stack, group, ori_data, mat in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            num_value_batch.append(num)
            group_batch.append(group)
            seg_batch.append(ori_data)
            ori_batch.append(' '.join(ori_data))
            mat_batch.append(pad_word_char_mat(mat, input_len_max, char_len_max))
        # max_num_len = max(num_size_batch)
        # n_broadcast_2_w = get_word_num_mat(input_len_max, num_pos_batch, max_num_len)

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        group_batches.append(group_batch)
        graph_batches.append(
            get_single_batch_graph(input_batch, input_length, group_batch, num_value_batch, num_pos_batch))
        mat_batches.append(mat_batch)
        seg_batches.append(seg_batch)
        ori_batches.append(ori_batch)


    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
           num_pos_batches, num_size_batches, num_value_batches, graph_batches, mat_batches, char_lengths, seg_batches, ori_batches


def prepare_ro_batch_KD(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    graph_batches = []
    num_value_batches = []
    mat_batches = []
    seg_batches = []
    ori_batches = []
    char_lengths = []
    teacher_batches=[]
    distill_weights=[]
    logit_batches=[]
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos + batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        char_length = []
        teacher_length=[]
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        # pair[2], pair[3], num_stack, pair[7], pair[5], mat, num_dict))
        for _, i, _, j, _, _, _, _, ori_data, mat,ts,logit in batch:
            input_length.append(i)
            output_length.append(j)
            for t,rank,score in ts:
                tlen=len(t)
                teacher_length.append(tlen)
            char_length.append(mat.shape[-1])
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        # output_len_max = max(output_length)
        output_len_max = max(output_length+teacher_length)
        char_len_max = max(char_length)
        char_lengths.append(char_length)

        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        group_batch = []
        num_value_batch = []
        seg_batch = []
        mat_batch = []
        ori_batch = []

        output_batch_teacher={"input":[],"inputLens":[],"target":[],"targetLens":[],
                              "num_stack":[],"num_pos":[],"num_size":[],"group":[],"num_value":[],
                              "mat":[],"seg":[],"char_length":[],"ori_data":[]}
        distill_weight_batch=[]
        logit_batch=[]
        
        for i, li, j, lj, num, num_pos, num_stack, group, ori_data, mat,teacher,logit in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            num_value_batch.append(num)
            group_batch.append(group)
            seg_batch.append(ori_data)
            ori_batch.append(' '.join(ori_data))
            mat_batch.append(pad_word_char_mat(mat, input_len_max, char_len_max))

            logit_batch.append(logit)
            teacher_out=[]
            teacher_len=[]
            weight=0
            decay=0.8
            # weight_sum=1+1/2+1/3+1/4+1/5
            weight_sum=sum(decay**r for r in range(5))
            
            ids=list(range(len(teacher)))
            if len(ids)>2:
                ids=random.sample(ids,2)
            for id,(tea,rank,score) in enumerate(teacher):
                # weight+=1/(rank+1)
                weight+=decay**rank

                seq_len=len(tea)
                if seq_len>output_len_max:
                    print("max len",seq_len,output_len_max)
                # teacher_out.append(pad_seq(tea,seq_len,output_len_max))
                # teacher_len.append(seq_len)
                if tuple(tea)!=tuple(j) and id in ids:
                    output_batch_teacher["input"].append(input_batch[-1])
                    output_batch_teacher["inputLens"].append(li)
                    output_batch_teacher["target"].append(pad_seq(tea, seq_len, output_len_max))
                    output_batch_teacher["targetLens"].append(seq_len)
                    output_batch_teacher["num_stack"].append(num_stack)
                    output_batch_teacher["num_pos"].append(num_pos)
                    output_batch_teacher["num_size"].append(len(num_pos))
                    output_batch_teacher["group"].append(group)
                    output_batch_teacher["num_value"].append(num)

                    output_batch_teacher["mat"].append(mat)
                    output_batch_teacher["seg"].append(ori_data)
                    output_batch_teacher["char_length"].append(mat.shape[-1])
                    output_batch_teacher['ori_data'].append(' '.join(ori_data))

            distill_weight_batch.append(weight/weight_sum)
        # max_num_len = max(num_size_batch)
        # n_broadcast_2_w = get_word_num_mat(input_len_max, num_pos_batch, max_num_len)

        
        if len(output_batch_teacher["input"])>0:
            char_len_max_t=max(output_batch_teacher["char_length"])
            input_len_max_t=max(output_batch_teacher["inputLens"])
            # print(char_len_max_t,char_len_max)
            output_batch_teacher["mat"]=[pad_word_char_mat(v, input_len_max_t, char_len_max_t) for v in output_batch_teacher["mat"]]
            output_batch_teacher["graph"]=get_single_batch_graph(output_batch_teacher["input"], 
                                                             output_batch_teacher["inputLens"],
                                                             output_batch_teacher["group"],
                                                             output_batch_teacher["num_value"],
                                                             output_batch_teacher["num_pos"])


        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        group_batches.append(group_batch)
        graph_batches.append(
            get_single_batch_graph(input_batch, input_length, group_batch, num_value_batch, num_pos_batch))
        mat_batches.append(mat_batch)
        seg_batches.append(seg_batch)
        ori_batches.append(ori_batch)

        teacher_batches.append(output_batch_teacher)
        distill_weights.append(distill_weight_batch)
        logit_batches.append(logit_batch)

    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
           num_pos_batches, num_size_batches, num_value_batches, graph_batches, mat_batches, char_lengths, seg_batches, ori_batches,\
           teacher_batches,distill_weights,logit_batches


def get_num_stack(eq, output_lang, num_pos):
    num_stack = []
    for word in eq:
        temp_num = []
        flag_not = True
        if word not in output_lang.index2word:
            flag_not = False
            for i, j in enumerate(num_pos):
                if j == word:
                    temp_num.append(i)
        if not flag_not and len(temp_num) != 0:
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:
            num_stack.append([_ for _ in range(len(num_pos))])
    num_stack.reverse()
    return num_stack


def prepare_de_train_batch(pairs_to_batch, batch_size, output_lang, rate, english=False):
    pairs = []
    b_pairs = copy.deepcopy(pairs_to_batch)
    for pair in b_pairs:
        p = copy.deepcopy(pair)
        pair[2] = check_bracket(pair[2], english)

        temp_out = exchange(pair[2], rate)
        temp_out = check_bracket(temp_out, english)

        p[2] = indexes_from_sentence(output_lang, pair[2])
        p[3] = len(p[2])
        pairs.append(p)

        temp_out_a = allocation(pair[2], rate)
        temp_out_a = check_bracket(temp_out_a, english)

        if temp_out_a != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out_a, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out_a)
            p[3] = len(p[2])
            pairs.append(p)

        if temp_out != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out)
            p[3] = len(p[2])
            pairs.append(p)

            if temp_out_a != pair[2]:
                p = copy.deepcopy(pair)
                temp_out_a = allocation(temp_out, rate)
                temp_out_a = check_bracket(temp_out_a, english)
                if temp_out_a != temp_out:
                    p[6] = get_num_stack(temp_out_a, output_lang, p[4])
                    p[2] = indexes_from_sentence(output_lang, temp_out_a)
                    p[3] = len(p[2])
                    pairs.append(p)
    print("this epoch training data is", len(pairs))
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        for i, li, j, lj, num, num_pos, num_stack in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches


# Multiplication exchange rate
def exchange(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    while idx < len(ex):
        s = ex[idx]
        if (s == "*" or s == "+") and random.random() < rate:
            lidx = idx - 1
            ridx = idx + 1
            if s == "+":
                flag = 0
                while not (lidx == -1 or ((ex[lidx] == "+" or ex[lidx] == "-") and flag == 0) or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex) or ((ex[ridx] == "+" or ex[ridx] == "-") and flag == 0) or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            else:
                flag = 0
                while not (lidx == -1
                           or ((ex[lidx] == "+" or ex[lidx] == "-" or ex[lidx] == "*" or ex[lidx] == "/") and flag == 0)
                           or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex)
                           or ((ex[ridx] == "+" or ex[ridx] == "-" or ex[ridx] == "*" or ex[ridx] == "/") and flag == 0)
                           or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            if lidx > 0 and ((s == "+" and ex[lidx - 1] == "-") or (s == "*" and ex[lidx - 1] == "/")):
                lidx -= 1
                ex = ex[:lidx] + ex[idx:ridx + 1] + ex[lidx:idx] + ex[ridx + 1:]
            else:
                ex = ex[:lidx] + ex[idx + 1:ridx + 1] + [s] + ex[lidx:idx] + ex[ridx + 1:]
            idx = ridx
        idx += 1
    return ex


def check_bracket(x, english=False):
    if english:
        for idx, s in enumerate(x):
            if s == '[':
                x[idx] = '('
            elif s == '}':
                x[idx] = ')'
        s = x[0]
        idx = 0
        if s == "(":
            flag = 1
            temp_idx = idx + 1
            while flag > 0 and temp_idx < len(x):
                if x[temp_idx] == ")":
                    flag -= 1
                elif x[temp_idx] == "(":
                    flag += 1
                temp_idx += 1
            if temp_idx == len(x):
                x = x[idx + 1:temp_idx - 1]
            elif x[temp_idx] != "*" and x[temp_idx] != "/":
                x = x[idx + 1:temp_idx - 1] + x[temp_idx:]
        while True:
            y = len(x)
            for idx, s in enumerate(x):
                if s == "+" and idx + 1 < len(x) and x[idx + 1] == "(":
                    flag = 1
                    temp_idx = idx + 2
                    while flag > 0 and temp_idx < len(x):
                        if x[temp_idx] == ")":
                            flag -= 1
                        elif x[temp_idx] == "(":
                            flag += 1
                        temp_idx += 1
                    if temp_idx == len(x):
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1]
                        break
                    elif x[temp_idx] != "*" and x[temp_idx] != "/":
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1] + x[temp_idx:]
                        break
            if y == len(x):
                break
        return x

    lx = len(x)
    for idx, s in enumerate(x):
        if s == "[":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == "]":
                    flag_b += 1
                elif x[temp_idx] == "[":
                    flag_b -= 1
                if x[temp_idx] == "(" or x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == "]" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "("
                x[temp_idx] = ")"
                continue
        if s == "(":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == ")":
                    flag_b += 1
                elif x[temp_idx] == "(":
                    flag_b -= 1
                if x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == ")" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "["
                x[temp_idx] = "]"
    return x


# Multiplication allocation rate
def allocation(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    lex = len(ex)
    while idx < len(ex):
        if (ex[idx] == "/" or ex[idx] == "*") and (ex[idx - 1] == "]" or ex[idx - 1] == ")"):
            ridx = idx + 1
            r_allo = []
            r_last = []
            flag = 0
            flag_mmd = False
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag += 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        r_last = ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                    elif ex[ridx] == "*" or ex[ridx] == "/":
                        flag_mmd = True
                        r_last = [")"] + ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                elif flag == -1:
                    r_last = ex[ridx:]
                    r_allo = ex[idx + 1: ridx]
                    break
                ridx += 1
            if len(r_allo) == 0:
                r_allo = ex[idx + 1:]
            flag = 0
            lidx = idx - 1
            flag_al = False
            flag_md = False
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag -= 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[lidx] == "+" or ex[lidx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                lidx -= 1
            if lidx != 0 and ex[lidx - 1] == "/":
                flag_al = False
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = lidx + 1
                temp_res = ex[:lidx]
                if flag_mmd:
                    temp_res += ["("]
                if lidx - 1 > 0:
                    if ex[lidx - 1] == "-" or ex[lidx - 1] == "*" or ex[lidx - 1] == "/":
                        flag_md = True
                        temp_res += ["("]
                flag = 0
                lidx += 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 0:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    temp_idx += 1
                temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo
                if flag_md:
                    temp_res += [")"]
                temp_res += r_last
                return temp_res
        if ex[idx] == "*" and (ex[idx + 1] == "[" or ex[idx + 1] == "("):
            lidx = idx - 1
            l_allo = []
            temp_res = []
            flag = 0
            flag_md = False  # flag for x or /
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag += 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[lidx] == "+":
                        temp_res = ex[:lidx + 1]
                        l_allo = ex[lidx + 1: idx]
                        break
                    elif ex[lidx] == "-":
                        flag_md = True  # flag for -
                        temp_res = ex[:lidx] + ["("]
                        l_allo = ex[lidx + 1: idx]
                        break
                elif flag == 1:
                    temp_res = ex[:lidx + 1]
                    l_allo = ex[lidx + 1: idx]
                    break
                lidx -= 1
            if len(l_allo) == 0:
                l_allo = ex[:idx]
            flag = 0
            ridx = idx + 1
            flag_al = False
            all_res = []
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag -= 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                ridx += 1
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = idx + 1
                flag = 0
                lidx = temp_idx + 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 1:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            all_res += l_allo + [ex[idx]] + ex[lidx: temp_idx] + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    if flag == 0:
                        break
                    temp_idx += 1
                if flag_md:
                    temp_res += all_res + [")"]
                elif ex[temp_idx + 1] == "*" or ex[temp_idx + 1] == "/":
                    temp_res += ["("] + all_res + [")"]
                temp_res += ex[temp_idx + 1:]
                return temp_res
        idx += 1
    return ex


