from src.expressions_transfer import from_infix_to_prefix


item={"num_exp": [
    "(",
    "(",
    "(",
    "N1",
    "-",
    "N0",
    ")",
    "/",
    "N0",
    ")",
    "*",
    "100",
    ")"
],
"nums": [
        "60",
        "150"
    ],
}

nums=item['nums']

pair=()
index2word=['/', '*', '-', '+', '^', 'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20', 'UNK']

prefix_exp=from_infix_to_prefix(item['num_exp'])
print(prefix_exp)

num_stack = []
for word in prefix_exp:
    temp_num = []
    flag_not = True
    if word not in index2word:
        print("word not in index2word:",word)
        flag_not = False
        # nums
        for i, j in enumerate(nums):
            if j == word:
                temp_num.append(i)
                print("temp:",i,j)

    if not flag_not and len(temp_num) != 0:
        num_stack.append(temp_num)
    if not flag_not and len(temp_num) == 0:
        num_stack.append([_ for _ in range(len(nums))])

print(num_stack)