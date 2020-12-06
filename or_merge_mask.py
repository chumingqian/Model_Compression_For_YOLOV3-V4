import torch


Merge_masks1 =torch.tensor( [ [1,1,0,1,1,0],[0,1,0,1,1,0],[0,0,0,1,1,0]] )
print(" \n the Merge_mask1: \n ", Merge_masks1)

step1= torch.sum(Merge_masks1,dim=0)
print(" \n the original step1: \n ",step1)

print(" \n the length of mask should be equal to = ", len(step1))

for  i  in range(len(step1)):
    print(f"  ***  ---the original  step1[{i}] = {step1[i]}  ")
    if  step1[i] < 3:
        step1[i] = 0
        print(f" **** --- the new  step1[{i}] = {step1[i]}  ")

print(f" \n after using  OR  func, the new  step1 = {step1}")



step2= ( step1 > 0)
print(" \n the step2: \n ",step2)




step3 = step2.float()
print(" \n the step3: \n ", step3)
