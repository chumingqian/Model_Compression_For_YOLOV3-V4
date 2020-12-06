import torch
'''
if len(Merge_masks) > 1:
    num_mask = len(Merge_masks)
    Merge_masks = torch.cat(Merge_masks, 0)
    step1 = torch.sum(Merge_masks, dim=0)
    for j in range(len(step1)):
        if step1[j] < (num_mask / 2):
            step1[j] = 0

    step2 = (step1 > 0)
    merge_mask = step2.float()
'''


Merge_masks =torch.tensor( [ [1,1,0,1,1,0],
                             [0,1,0,1,1,0],
                             [0,0,0,1,1,0],
                             [0,0,0,0,0,0],
                             [1,0,0,0,0,0],
                             ] )
print(" \n the Merge_mask1: \n ", Merge_masks)

# 统计出 Merge_masks 中 每个mask， 所包含1的个数
# 取出其中 1的个数 最少的mask
# 并找出在Merge_masks 中 对应的下标；
#  将此下标的 对应的mask  --> 作为在这个模块中，所有shortcut相关联的卷积层的 最终统一的merge_mask;
nums_mask = []
for mask_index  in range(len(Merge_masks)):
    print( f"Merge_masks[{mask_index}]  =  {Merge_masks[mask_index]}")

    num_ofOne = Merge_masks[mask_index].sum()
    nums_mask.append(num_ofOne)

temp = torch.Tensor(nums_mask)
print(f'temp = {temp}')

least_index = temp.argmin()
print(f' least_index = {least_index}')

temp_merge_mask = Merge_masks[least_index]
print(f'in this module, the final merge_mask = {merge_mask} ')

# 使用 torch.squeeze(a)函数， 压缩tensor a中 维数是 1 的那个维度；
#  减少掉 维数 是1 的 那个维度；
merge_mask = torch.squeeze(temp_merge_mask)