dataset = 'dblp'
ground_truth = dict()
with open('./ground_truth.txt', "r") as f:
    lines = f.readlines()
    for line in lines:
        temp = list(line.strip('\n').split('\t'))
        ground_truth[temp[0]] = temp[1]
pred1 = dict()
with open('./dblp_pred1', "r") as f:
    lines = f.readlines()
    for line in lines:
        temp = list(line.strip('\n').split('\t'))
        pred1[temp[0]] = temp[1]
# pred2 = dict()
# with open('./data/' + dataset + '/rank/pred2', "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         temp = list(line.strip('\n').split('\t'))
#         pred2[temp[0]] = temp[1]
# pred3 = dict()
# with open('./data/' + dataset + '/rank/pred3', "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         temp = list(line.strip('\n').split('\t'))
#         pred3[temp[0]] = temp[1]
# pred4 = dict()
# with open('./data/' + dataset + '/rank/pred4', "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         temp = list(line.strip('\n').split('\t'))
#         pred4[temp[0]] = temp[1]
# pred5 = dict()
# with open('./data/' + dataset + '/rank/pred5', "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         temp = list(line.strip('\n').split('\t'))
#         pred5[temp[0]] = temp[1]


def hit_num(gt, pd):
    h_num = 0
    for l in pd.keys():
        if gt[l] == pd[l]:
            h_num += 1
    return h_num


hit_num1 = hit_num(ground_truth, pred1)
# hit_num2 = hit_num(ground_truth, pred2)
# hit_num3 = hit_num(ground_truth, pred3)
# hit_num4 = hit_num(ground_truth, pred4)
# hit_num5 = hit_num(ground_truth, pred5)

hit_at_1 = (hit_num1) / (len(pred1))
print(hit_at_1)
