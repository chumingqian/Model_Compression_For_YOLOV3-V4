from models import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse


# %%
def slim_obtain_filters_mask(model, thre, CBL_idx, prune_idx):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:

            weight_copy = bn_module.weight.data.abs().clone()

            channels = weight_copy.shape[0]  #
            min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 16
            mask = weight_copy.gt(thresh).float()

            if int(torch.sum(mask)) < min_channel_num:
                _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                mask[sorted_index_weights[:min_channel_num]] = 1.
            remain = int(mask.sum())
            pruned = pruned + mask.shape[0] - remain

            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d}')
        else:
            mask = torch.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]

        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.clone())

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_filters, filters_mask


def twoPart_slim_obtain_filters_mask(model, backbone_thre, truncated_layer,non_backbone_thre, CBL_idx, prune_idx):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []

    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]

        if idx <  truncated_layer:

            if idx in prune_idx:

                weight_copy = bn_module.weight.data.abs().clone()

                channels = weight_copy.shape[0]  #
                min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 16
                mask = weight_copy.gt(backbone_thre).float()

                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

            if idx + 1 == truncated_layer or idx + 2 == truncated_layer:
                print(f' \n ×××××  Here we truncated  from this truncated_layer = {truncated_layer} ****** \n ')


        elif  idx >=   truncated_layer :


            if idx - 1 == truncated_layer or idx - 2 == truncated_layer:
                print(f' \n ×××××  Here we truncated  from this truncated_layer = {truncated_layer} ****** \n ')


            if idx in prune_idx:

                weight_copy = bn_module.weight.data.abs().clone()

                channels = weight_copy.shape[0]  #
                min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 16
                mask = weight_copy.gt(non_backbone_thre).float()

                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.clone())

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_filters, filters_mask



def allGroups_slim_obtain_filters_mask(model, truncated_layer,groups_thre, CBL_idx, prune_idx):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []

    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]

        if idx <  truncated_layer[0]:

            if idx in prune_idx:

                weight_copy = bn_module.weight.data.abs().clone()

                channels = weight_copy.shape[0]  #
                min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 16
                mask = weight_copy.gt(groups_thre[0]).float()

                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

            if idx + 1 == truncated_layer[0] or idx + 2 == truncated_layer[0]:
                print(f' \n ××××× After Here we truncated  from this truncated_layer = {truncated_layer[0]} ****** \n ')


        elif   truncated_layer[0] <=  idx   and   idx  < truncated_layer[1] :


            if idx - 1 == truncated_layer[0] or idx - 2 == truncated_layer[0]:
                print(f' \n ××××× Group2 Here we truncated  from this truncated_layer = {truncated_layer[0]} ****** \n ')


            if idx in prune_idx:

                weight_copy = bn_module.weight.data.abs().clone()

                channels = weight_copy.shape[0]  #
                min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 16
                mask = weight_copy.gt(groups_thre[1]).float()

                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]


        elif truncated_layer[1] <= idx and idx < truncated_layer[2]:

            if idx - 1 == truncated_layer[1] or idx - 2 == truncated_layer[1]:
                print(f' \n ××××× Group3  Here we truncated  from this truncated_layer = {truncated_layer[1]} ****** \n ')

            if idx in prune_idx:

                weight_copy = bn_module.weight.data.abs().clone()

                channels = weight_copy.shape[0]  #
                min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 16
                mask = weight_copy.gt(groups_thre[2]).float()

                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

        elif truncated_layer[2] <= idx and idx < truncated_layer[3]:

            if idx - 1 == truncated_layer or idx - 2 == truncated_layer:
                print(f' \n ××××× Group4 Here we truncated  from this truncated_layer = {truncated_layer[2]} ****** \n ')

            if idx in prune_idx:

                weight_copy = bn_module.weight.data.abs().clone()

                channels = weight_copy.shape[0]  #
                min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 16
                mask = weight_copy.gt(groups_thre[3]).float()

                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]


        elif truncated_layer[3] <= idx and idx < truncated_layer[4]:

            if idx - 1 == truncated_layer[3] or idx - 2 == truncated_layer[3]:
                print(f' \n ×××××  Group5  Here we truncated  from this truncated_layer = {truncated_layer[3]} ****** \n ')

            if idx in prune_idx:

                weight_copy = bn_module.weight.data.abs().clone()

                channels = weight_copy.shape[0]  #
                min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 16
                mask = weight_copy.gt(groups_thre[4]).float()

                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]


        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.clone())

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_filters, filters_mask


def prune_and_eval(model, CBL_idx, CBLidx2mask):
    model_copy = deepcopy(model)

    for idx in CBL_idx:
        bn_module = model_copy.module_list[idx][1]
        mask = CBLidx2mask[idx].cuda()
        bn_module.weight.data.mul_(mask)

    with torch.no_grad():
        mAP = eval_model(model_copy)[0][2]

    print(f'mask the gamma as zero, mAP of the model is {mAP:.4f}')




def obtain_avg_forward_time(input, model, repeat=200):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.8, help='global channel prune percent')
    parser.add_argument('--layer_keep', type=float, default=0.01, help='channel keep percent per layer')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--truncated_layer', type=int, default=118, help='the first truncated layer')

    opt = parser.parse_args()
    percent = opt.percent
    truncated_layer = opt.truncated_layer

    print(opt)

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("\n  Original model and it's  parameters number : \n")
    print(" The Darknet()  class  will  print the model information \n")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, opt.weights)
    print('\nloaded weights from ', opt.weights)

    eval_model = lambda model: test(model=model, cfg=opt.cfg, data=opt.data, batch_size=8, imgsz=img_size)
    obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])

    print("\n testing the original model's performance :")
    with torch.no_grad():
        origin_model_metric = eval_model(model)
    origin_nparameters = obtain_num_parameters(model)
    # CBL_idx, Other_idx, prune_idx, shortcut_idx, shortcut_all
    CBL_idx, Conv_idx, prune_idx, shortcut_idx, _ = shortcut_parse_module_defs(model.module_defs)

    bn_weights = gather_bn_weights(model.module_list, prune_idx)
    print(" \n the total bn_weights which are the number of channel  in the  CBL_layers can be pruned  are = ",
          len(bn_weights))

    truncated_layer = [56, 86, 106, 136, 162]



    group1_bn_weights, group2_bn_weights,group3_bn_weights,group4_bn_weights,group5_bn_weights = gather_all_group_bn_weights(model.module_list, prune_idx,
                                                                             truncated_layer)





    print("\n  NOW, checking if  the total number  of all separete groups bn_wieght  ==  the original  total  number of bn_weight  ")

    already_get_num = len(group1_bn_weights) +  len(group2_bn_weights) + len(group3_bn_weights)  + len(group4_bn_weights) + len(group5_bn_weights)

    if  already_get_num == len(bn_weights):
         print(" the  first sparte groups  this  steop   tempolary  pass  pass  \n")

    else:
        print(" the  first ,sparte groups  this  step  pass \n")
    assert already_get_num == len(bn_weights)


    print(" \n  group1,   numbers of backbone_bn_weights can be pruned  are = ", len(group1_bn_weights))
    print(" \n  group2,   numbers of backbone_bn_weights can be pruned  are = ", len(group2_bn_weights))
    print(" \n  group3,   numbers of backbone_bn_weights can be pruned  are = ", len(group3_bn_weights))
    print(" \n  group4,   numbers of backbone_bn_weights can be pruned  are = ", len(group4_bn_weights))
    print(" \n  group5,   numbers of backbone_bn_weights can be pruned  are = ", len(group5_bn_weights))





    sorted_bn = torch.sort(bn_weights)[0]


    g1_sorted_bn = torch.sort(group1_bn_weights)[0]
    g2_sorted_bn = torch.sort(group2_bn_weights)[0]
    g3_sorted_bn = torch.sort(group3_bn_weights)[0]
    g4_sorted_bn = torch.sort(group4_bn_weights)[0]
    g5_sorted_bn = torch.sort(group5_bn_weights)[0]




    '''
    sorted_bn, sorted_index = torch.sort(bn_weights)
    thresh_index = int(len(bn_weights) * opt.percent)
    thresh = sorted_bn[thresh_index].cuda()

    print(f'Global Threshold should be less than {thresh:.15f}.')
    '''


    g1_pruned_percent = 0.15
    g2_pruned_percent = 0.32
    g3_pruned_percent = 0.91
    g4_pruned_percent = 0.91
    g5_pruned_percent = 0.90

    print(f' \n *****\n Part1: In the group1 part,  group1_pruned_percent= {g1_pruned_percent:.10f} are pruned!\n')
    print(f'Part2: In the group2 part,  group2_pruned_percent= {g2_pruned_percent:.10f} are pruned!\n')
    print(f'Part2: In the group3 part,  group3_pruned_percent= {g3_pruned_percent:.10f} are pruned!\n')
    print(f'Part2: In the group4 part,  group4_pruned_percent= {g4_pruned_percent:.10f} are pruned!\n')
    print(f'Part2: In the group5 part,  group5_pruned_percent= {g5_pruned_percent:.10f} are pruned!\n ***** \n')





    g1_thre_index = int(len(g1_sorted_bn) * g1_pruned_percent)
    g2_thre_index = int(len(g2_sorted_bn) * g2_pruned_percent)
    g3_thre_index = int(len(g3_sorted_bn) * g3_pruned_percent)
    g4_thre_index = int(len(g4_sorted_bn) * g4_pruned_percent)
    g5_thre_index = int(len(g5_sorted_bn) * g5_pruned_percent)

    # 获得α参数的阈值，小于该值的α参数对应的通道，全部裁剪掉
    g1_thre = g1_sorted_bn[g1_thre_index]
    g2_thre = g2_sorted_bn[g2_thre_index]
    g3_thre = g3_sorted_bn[g3_thre_index]
    g4_thre = g4_sorted_bn[g4_thre_index]
    g5_thre = g5_sorted_bn[g5_thre_index]

    print(f' Part1: In the group1 part,  Channels with Gamma value less than {g1_thre:.15f} are pruned!\n')
    print(f' Part1: In the group2 part,  Channels with Gamma value less than {g2_thre:.15f} are pruned!\n')
    print(f' Part1: In the group3 part,  Channels with Gamma value less than {g3_thre:.15f} are pruned!\n')
    print(f' Part1: In the group4 part,  Channels with Gamma value less than {g4_thre:.15f} are pruned!\n')
    print(f' Part1: In the group5 part,  Channels with Gamma value less than {g5_thre:.15f} are pruned!\n')


    groups_thre = [ g1_thre, g2_thre, g3_thre, g4_thre, g5_thre]

    # num_filters, filters_mask = slim_obtain_filters_mask(model, thresh, CBL_idx, prune_idx)
    # num_filters, filters_mask = twoPart_slim_obtain_filters_mask(model, backbone_thre,truncated_layer,non_backbone_thre, CBL_idx, prune_idx)
    num_filters, filters_mask = allGroups_slim_obtain_filters_mask(model,truncated_layer,groups_thre, CBL_idx, prune_idx)




    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    CBLidx2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)}

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i['is_access'] = False

    print('\n using Least_Mask_merge the mask of layers connected to shortcut!:')
    or_merge_mask(model, CBLidx2mask, CBLidx2filters)

    prune_and_eval(model, CBL_idx, CBLidx2mask)

    for i in CBLidx2mask:
        CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()

    pruned_model = prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask)
    print("\nnow prune the model but keep size,(actually add offset of BN beta to following layers), let's see how the mAP goes")

    with torch.no_grad():
        eval_model(pruned_model)

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i.pop('is_access')

    compact_module_defs = deepcopy(model.module_defs)
    for idx in CBL_idx:
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(CBLidx2filters[idx])

    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
    compact_nparameters = obtain_num_parameters(compact_model)

    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    print('testing inference time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    print('testing the final model...')
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model)

    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{origin_model_metric[0][2]:.6f}', f'{compact_model_metric[0][2]:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)

    pruned_cfg_name = opt.cfg.replace('/', f'/group_slim_OrMask_prune_{opt.percent}_keep_{opt.layer_keep}_')
    # 创建存储目录
    dir_name = pruned_cfg_name.split('/')[0] + '/' + pruned_cfg_name.split('/')[1]
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # 由于原始的compact_module_defs将anchors从字符串变为了数组，因此这里将anchors重新变为字符串
    file = open(opt.cfg, 'r')
    lines = file.read().split('\n')
    for line in lines:
        if line.split(' = ')[0] == 'anchors':
            anchor = line.split(' = ')[1]
            break
    file.close()

    for item in compact_module_defs:
        if item['type'] == 'shortcut':
            item['from'] = str(item['from'][0])
        elif item['type'] == 'route':
            item['layers'] = ",".join('%s' % i for i in item['layers'])
        elif item['type'] == 'yolo':
            item['mask'] = ",".join('%s' % i for i in item['mask'])
            item['anchors'] = anchor
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print(f'Config file has been saved: {pruned_cfg_file}')

    # chu Y add
    # to distingush yolov3 yolov4 name

    # compact_model_name = opt.weights.replace('/', f'/prune_{opt.percent}_keep_{opt.layer_keep}_')
    compact_model_name = opt.weights.replace(opt.weights.split('/')[-1], f'group_slim_OrMask_prune_{opt.percent}_layer_keep_{opt.layer_keep}.weights')

    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    save_weights(compact_model, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')
