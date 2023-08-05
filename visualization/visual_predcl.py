# %%
import matplotlib.pyplot as plt
import torch
import json
import h5py
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
# %%
# -------------------------!!!GCL的要重排序!!!------------------------------
predicate_new_order_name = {'1':'on', '2':'has', '3':'wearing', '4':'of', '5':'in', '6':'near', '7':'behind', '8':'with', '9':'holding', '10':'above',
                            '11':'sitting on', '12':'wears', '13':'under', '14':'riding', '15':'in front of', '16':'standing on', '17':'at', '18':'carrying', '19':'attached to', '20':'walking on',
                            '21':'over', '22':'for', '23':'looking at', '24':'watching', '25':'hanging from', '26':'laying on', '27':'eating', '28':'and', '29':'belonging to', '30':'parked on',
                            '31':'using', '32':'covering', '33':'between', '34':'along', '35':'covered in', '36':'part of', '37':'lying on', '38':'on back of', '39':'to', '40':'walking in',
                            '41':'mounted on', '42':'across', '43':'against', '44':'from', '45':'growing on', '46':'painted on', '47':'playing', '48':'made of', '49':'says', '50':'flying in'}

# image_file = json.load(open('/home/kaihua/projects/maskrcnn-benchmark/datasets/vg/image_data.json'))
# vocab_file = json.load(open('/home/kaihua/projects/maskrcnn-benchmark/datasets/vg/VG-SGG-dicts.json'))
# data_file = h5py.File('/home/kaihua/projects/maskrcnn-benchmark/datasets/vg/VG-SGG.h5', 'r')
image_file = json.load(open('/home/stormai/userfile/zrn/datasets/image_data.json'))
vocab_file = json.load(open('/home/stormai/userfile/zrn/datasets/VG-SGG-dicts-with-attri.json'))
data_file = h5py.File('/home/stormai/userfile/zrn/datasets/VG-SGG-with-attri.h5', 'r')
# remove invalid image
corrupted_ims = [1592, 1722, 4616, 4617]
tmp = []
for item in image_file:
    if int(item['image_id']) not in corrupted_ims:
        tmp.append(item)
image_file = tmp

# %%

# load detected results
# detected_origin_path = '/home/kaihua/checkpoints/vctree-sgcls-only-vis/inference/VG_stanford_filtered_with_attribute_test/'
detected_origin_path = '/home/stormai/userfile/zrn/SHA-GCL/tools/output/relation_baseline/inference_final/ours-VG_stanford_filtered_with_attribute_test/'
detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')
detected_info = json.load(open(detected_origin_path + 'visual_info.json'))

# %%

# get image info by index
def get_info_by_idx(idx, det_input, thres=0.5):
    groundtruth = det_input['groundtruths'][idx]
    prediction = det_input['predictions'][idx]
    # image path
    img_path = detected_info[idx]['img_file']
    print(img_path)
    # boxes
    boxes = groundtruth.bbox
    # object labels
    idx2label = vocab_file['idx_to_label']
    labels = ['{}-{}'.format(idx, idx2label[str(i)]) for idx, i in enumerate(groundtruth.get_field('labels').tolist())]
    pred_labels = ['{}-{}'.format(idx, idx2label[str(int(i))]) for idx, i in
                   enumerate(prediction.get_field('pred_labels').tolist())]
    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    gt_rels = groundtruth.get_field('relation_tuple').tolist()
    gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    pred_rel_label = prediction.get_field('pred_rel_scores')
    pred_rel_label[:, 0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
    # mask = pred_rel_score > thres
    # pred_rel_score = pred_rel_score[mask]
    # pred_rel_label = pred_rel_label[mask]
    # 原顺序
    # pred_rels = [(pred_labels[int(i[0])], idx2pred[str(j)], pred_labels[int(i[1])]) for i, j in
    #              zip(pred_rel_pair, pred_rel_label.tolist())]
    pred_rels = [(pred_labels[int(i[0])], predicate_new_order_name[str(j)], pred_labels[int(i[1])]) for i, j in
                 zip(pred_rel_pair, pred_rel_label.tolist())]
    return img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label

# gt:ear-on-head,预测的谓语不是on而是其他的谓语，统计数量
def find_gtis_relnot(det_input):
    groundtruth = det_input['groundtruths']
    prediction = det_input['predictions']
    img_num = len(groundtruth)
    idx2label = vocab_file['idx_to_label']
    idx2pred = vocab_file['idx_to_predicate']
    rec_dic = {}

    for gt_i, pred_i in zip(groundtruth, prediction):
        labels = ['{}-{}'.format(idx, idx2label[str(i)]) for idx, i in enumerate(gt_i.get_field('labels').tolist())]
        pred_labels = ['{}-{}'.format(idx, idx2label[str(int(i))]) for idx, i in enumerate(pred_i.get_field('pred_labels').tolist())]
        gt_rels = gt_i.get_field('relation_tuple').tolist()
        gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
        pred_rel_pair = pred_i.get_field('rel_pair_idxs').tolist()
        pred_rel_label = pred_i.get_field('pred_rel_scores')
        pred_rel_label[:, 0] = 0
        pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
        pred_rels = [(pred_labels[int(i[0])], idx2pred[str(j)], pred_labels[int(i[1])]) for i, j in
                     zip(pred_rel_pair, pred_rel_label.tolist())]
        for j in gt_rels:
            true_j = predicate_new_order_name[str(ori_order[j[1]])] # 调整顺序
            if true_j == 'behind':
                for k in pred_rels:
                    if (j[0] == k[0]) and (j[2]==k[2]):
                        if k[1] in rec_dic.keys():
                            rec_dic[k[1]] += 1
                            break # 当前gt的主客体对已经不需要再验证了
                        else:
                            new_key = k[1]
                            rec_dic[new_key] = 1
                            break # 当前gt的主客体对已经不需要再验证了

    dic_sum = sum(rec_dic.values())
    for k,v in rec_dic.items():
        rec_dic[k] = rec_dic[k] / dic_sum

    final_rec_dic = {}
    for k, v in rec_dic.items():
        true_k = predicate_new_order_name[str(ori_order[k])]
        final_rec_dic[true_k] = v

    return final_rec_dic
ori_order = vocab_file['predicate_to_idx']
final_rec_dic = find_gtis_relnot(detected_origin_result)
print()
# %%

# ground truth在左边竖着 预测的谓语在右边横着
def heat(det_input):
    groundtruth = det_input['groundtruths']
    prediction = det_input['predictions']
    img_num = len(groundtruth)
    idx2label = vocab_file['idx_to_label']
    idx2pred = vocab_file['idx_to_predicate']
    rec_dic = {}
    rec_num = {}
    for gt_i, pred_i in zip(groundtruth, prediction):
        labels = ['{}-{}'.format(idx, idx2label[str(i)]) for idx, i in enumerate(gt_i.get_field('labels').tolist())]
        pred_labels = ['{}-{}'.format(idx, idx2label[str(int(i))]) for idx, i in enumerate(pred_i.get_field('pred_labels').tolist())]
        gt_rels = gt_i.get_field('relation_tuple').tolist()
        gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
        pred_rel_pair = pred_i.get_field('rel_pair_idxs').tolist()
        pred_scores = pred_i.get_field('pred_rel_scores')
        pred_scores[:, 0] = 0
        pred_rel_label = pred_i.get_field('pred_rel_scores')
        pred_rel_label[:, 0] = 0
        pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
        pred_rels = [(pred_labels[int(i[0])], idx2pred[str(j)], pred_labels[int(i[1])]) for i, j in
                     zip(pred_rel_pair, pred_rel_label.tolist())]

        for j in gt_rels:
            tmp_dic = {}
            tmp_num = {}
            if 'table' in j[0] and 'chair' in j[2]:
                if j[1] in rec_dic.keys():
                    tmp_dic = rec_dic[j[1]]
                    tmp_num = rec_num[j[1]]
                else:
                    new_key = j[1]
                    rec_dic[new_key] = {}
                    rec_num[new_key] = {}

                for k, s in zip(pred_rels, pred_scores):
                    if (j[0] == k[0]) and (j[2]==k[2]):
                        if k[1] in tmp_dic.keys():
                            tmp_dic[k[1]] += s
                            tmp_num[k[1]] += 1
                            break # 当前gt的主客体对已经不需要再验证了
                        else:
                            new_key = k[1]
                            tmp_dic[new_key] = s
                            tmp_num[new_key] = 1
                            break # 当前gt的主客体对已经不需要再验证了
                rec_dic[j[1]] = tmp_dic
                rec_num[j[1]] = tmp_num

    # row_sum = []
    # for k,v in rec_dic.items():
    #     r_sum = sum(rec_dic[k].values())
    #     row_sum.append(r_sum)

    rec_list  = []
    for k, v in rec_dic.items():
        for kk, vv in v.items():
            rec_dic[k][kk] /= rec_num[k][kk]

    for k, v in rec_dic.items():
        a = sum(rec_dic[k].values())/len(rec_dic[k])
        rec_dic[k] = a

    label2idx = vocab_file['predicate_to_idx'] # 从谓语名称转为id，即列下标
    pred_cols = []
    for k in rec_dic.keys():
        col = label2idx[k]
        pred_cols += [col]

    for k, v in rec_dic.items(): # matrix
        score = rec_dic[k][pred_cols]
        rec_dic[k] = score
    ori_order = vocab_file['predicate_to_idx']
    # 把最终的顺序换回来 即正确的谓语名称
    final_rec_dic = {}
    for k, v in rec_dic.items():
        true_k = predicate_new_order_name[str(ori_order[k])]
        final_rec_dic[true_k] = v

    return final_rec_dic

# rec_dic = heat(detected_origin_result)
print()

def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    # x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)

def print_list(name, input_list):
    for i, item in enumerate(input_list):
        print(name + ' ' + str(i) + ': ' + str(item))

def draw_image(img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label, print_img=True):
    pic = Image.open(img_path)
    num_obj = boxes.shape[0]
    for i in range(num_obj):
        info = labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
    if print_img:
        plt.axis('off') # 不显示坐标轴
        plt.imshow(pic)
        plt.show()
        Image._show(pic)
        # display(pic) #
    if print_img:
        print('*' * 50)
        print_list('gt_boxes', labels)
        print('*' * 50)
        print_list('gt_rels', gt_rels)
        print('*' * 50)
    print_list('pred_rels', pred_rels)
    print('*' * 50)

    return None

# %%

def show_selected(idx_list):
    for select_idx in idx_list:
        print(select_idx)
        draw_image(*get_info_by_idx(select_idx, detected_origin_result))


def show_all(start_idx, length):
    for cand_idx in range(start_idx, start_idx + length):
        print(cand_idx)
        draw_image(*get_info_by_idx(cand_idx, detected_origin_result))


# %%
# show_all(start_idx=0, length=5)
# show_selected([2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499]) #
show_selected([2522,2523,2524,2525,2526])