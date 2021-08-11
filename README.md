# Model_Compression_For_YOLOV4
In this  repository  using the sparse training, group pruning and  knowledge distilling for  YOLOV3 and YOLOV4


# YOLOv3v4 -ModelCompression-Training

This project mainly include three parts.

Part1.  Support a common training and sparse training(prepare for the channel pruning) for three object detection datasets(COCO2017, VOC, OxforHand)

Part2.  Support a general model compression algorithm including pruning and knowledge distillation.

Part3.  Support for yolov4 and yolov3 netwrok.

Source using Pytorch implementation to [ultralytics/yolov3](https://github.com/ultralytics/yolov3) for yolov3 source code.

For the  YOLOV4 pytorch version, try this https://github.com/Tianxiaomo/pytorch-YOLOv4.

---------

### Datasets and  Environment Requirements
Make a COCO or VOC dataset for this project try here [dataset_for_Ultralytics_training](https://github.com/chumingqian/Make_Dataset-for-Ultralytics-yolov3v4).

The environment is Pytorch >= 1.1.0 , see the ./requiremnts.txt and also we can see [ultralytics/yolov3](https://github.com/ultralytics/yolov3) ./requirements.txt .

  
###  Part1.Common training and sparse training(prepare for the channel pruning) training for object detection datasets

1.1 For the common training use the following command: 

   `python3 train.py --data ...  --cfg ...  -pt  --weights ...  --img_size ... --batch-size ... --epochs  ... ` the -pt means that will  use the pretrained  model's weight.

1.2 For the sparse training use the:
```bash
python3 train.py --data ... --s 0.001 --prune 0  -pt --weights ... --cfg ... --img_size ...  --batch-size 32  --epochs ...
```

1.3 parameter explaination:

`-sr`Sparse training,`--s`Specifies the sparsity factor，`--prune`Specify the sparsity type.

`--prune 0` is the sparsity of normal pruning and regular pruning.

`--prune 1` is the sparsity of shortcut pruning.

`--prune 2` is the sparsity of layer pruning.

1.4 Notice for the sparse training, the reason for using sparse training before we prune the network  is that  we need to select out the unimportant channels in the network, through the sparse training we can select out and prune  these unimportant channels  in the network. 
  There  maybe  no   when the  classes  you trian is  not too much ,such 1-5 classes.
```bash
python3 train.py --data ... --s 0.001 --prune 0  -pt --weights ... --cfg ... --img_size ...  --batch-size 32  --epochs ...
```

1.3 Testing and detect the modle
`python3 test.py --data ... --cfg ... ` Test command

`python3 detect.py --data ... --cfg ... --source ...` Detection command, the default address of source is data/samples, the output result is saved in the output file, and the detection resource can be pictures, videos.



# Model Compression

## 1. Pruning

### channel pruning types 
|<center>method</center> |<center>advantage</center>|<center>disadvantage</center> |
| --- | --- | --- |
|Normal pruning        |Not prune for shortcut layer. It has a considerable and stable compression rate that requires no fine tuning.|The compression rate is not extreme.  |
|Shortcut pruning      |Very high compression rate.  |Fine-tuning is necessary.  |
|Silmming              |Shortcut fusion method was used to improve the precision of shear planting.|Best way for shortcut pruning|
|Regular pruning       |Designed for hardware deployment, the number of filters after pruning is a multiple of 2, no fine-tuning, support tiny-yolov3 and Mobilenet series.|Part of the compression ratio is sacrificed for regularization. |
|layer pruning         |ResBlock is used as the basic unit for purning, which is conducive to hardware deployment. |It can only cut backbone. |
|layer-channel pruning |First, use channel pruning and then use layer pruning, and pruning rate was very high. |Accuracy may be affected. |

### Step


2.Sparse training

`-sr`Sparse training,`--s`Specifies the sparsity factor，`--prune`Specify the sparsity type.

`--prune 0` is the sparsity of normal pruning and regular pruning.

`--prune 1` is the sparsity of shortcut pruning.

`--prune 2` is the sparsity of layer pruning.

command：

3.Pruning

- normal pruning
```bash
python3 normal_prune.py --cfg ... --data ... --weights ... --percent ...
```
- regular pruning
```bash
python3 regular_prune.py --cfg ... --data ... --weights ... --percent ...
```
- shortcut pruning
```bash
python3 shortcut_prune.py --cfg ... --data ... --weights ... --percent ...
```

- silmming
```bash
python3 slim_prune.py --cfg ... --data ... --weights ... --percent ...
```

- layer pruning
```bash
python3 layer_prune.py --cfg ... --data ... --weights ... --shortcut ...
```

- layer-channel pruning
```bash
python3 layer_channel_prune.py --cfg ... --data ... --weights ... --shortcut ... --percent ...
```


It is important to note that the cfg and weights variables in OPT need to be pointed to the cfg and weights files generated by step 2.

In addition, you can get more compression by increasing the percent value in the code.
(If the sparsity is not enough and the percent value is too high, the program will report an error.)

### Pruning experiment
1.normal pruning
oxfordhand，img_size = 608，test on GTX2080Ti*4

|<center>model</center> |<center>parameter before pruning</center> |<center>mAP before pruning</center>|<center>inference time before pruning</center>|<center>percent</center> |<center>parameter after pruning</center> |<center>mAP after pruning</center> |<center>inference time after pruning</center>
| --- | --- | --- | --- | --- | --- | --- | --- |
|yolov3(without fine tuning)     |58.67M   |0.806   |0.1139s   |0.8    |10.32M |0.802 |0.0844s |
|yolov3-mobilenet(fine tuning)   |22.75M   |0.812   |0.0345s   |0.97   |2.72M  |0.795 |0.0211s |
|yolov3tiny(fine tuning)         |8.27M    |0.708   |0.0144s   |0.5    |1.13M  |0.641 |0.0116s |

2.regular pruning
oxfordhand，img_size = 608，test ong GTX2080Ti*4

|<center>model</center> |<center>parameter before pruning</center> |<center>mAP before pruning</center>|<center>inference time before pruning</center>|<center>percent</center> |<center>parameter after pruning</center> |<center>mAP after pruning</center> |<center>inference time after pruning</center>
| --- | --- | --- | --- | --- | --- | --- | --- |
|yolov3(without fine tuning)           |58.67M   |0.806   |0.1139s   |0.8    |12.15M |0.805 |0.0874s |
|yolov3-mobilenet(fine tuning)   |22.75M   |0.812   |0.0345s   |0.97   |2.75M  |0.803 |0.0208s |
|yolov3tiny(fine tuning)         |8.27M    |0.708   |0.0144s   |0.5    |1.82M  |0.703 |0.0122s |

## 2.quantization

`--quantized 2` Dorefa quantization method

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --quantized 2
```

`--quantized 1` Google quantization method

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --quantized 3
```

`--BN_Flod` using BN Flod training, `--FPGA` Pow(2) quantization for FPGA.
### experiment
oxfordhand, yolov3, 640image-size
|<center>method</center> |<center>mAP</center> |
| --- | --- |
|Baseline                     |0.847    |
|Google8bit                   |0.851    |
|Google8bit + BN Flod         |0.851    |
|Google8bit + BN Flod + FPGA  |0.852    |
|Google4bit + BN Flod + FPGA  |0.842    |
## 3.Knowledge Distillation

### Knowledge Distillation
The distillation method is based on the basic distillation method proposed by Hinton in 2015, and has been partially improved in combination with the detection network.

Distilling the Knowledge in a Neural Network
[paper](https://arxiv.org/abs/1503.02531)

command : `--t_cfg --t_weights --KDstr` 

`--t_cfg` cfg file of teacher model

`--t_weights` weights file of teacher model

`--KDstr` KD strategy

    `--KDstr 1` KLloss can be obtained directly from the output of teacher network and the output of student network and added to the overall loss.
    `--KDstr 2` To distinguish between box loss and class loss, the student does not learn directly from the teacher. L2 distance is calculated respectively for student, teacher and GT. When student is greater than teacher, an additional loss is added for student and GT.
    `--KDstr 3` To distinguish between Boxloss and ClassLoss, the student learns directly from the teacher.
    `--KDstr 4` KDloss is divided into three categories, box loss, class loss and feature loss.
    `--KDstr 5` On the basis of KDstr 4, the fine-grain-mask is added into the feature

example:

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --t_cfg ... --t_weights ...
```

Usually, the pre-compression model is used as the teacher model, and the post-compression model is used as the student model for distillation training to improve the mAP of student network.

### experiment
oxfordhand，yolov3tiny as teacher model，normal pruning yolov3tiny as student model

|<center>teacher model</center> |<center>mAP of teacher model</center> |<center>student model</center>|<center>directly fine tuning</center>|<center>KDstr 1</center> |<center>KDstr 2</center> |<center>KDstr 3</center>  |<center>KDstr 4(L1)</center> |<center>KDstr 5(L1)</center> |
| --- | --- | --- | --- | --- | --- | --- |--- |--- |
|yolov3tiny608   |0.708    |normal pruning yolov3tiny608    |0.658     |0.666    |0.661  |0.672   |0.673   |0.674   |



## Reference: 
----

Pruning method based on BN layer comes from [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519).

Pruning without fine-tune [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/pdf/1802.00124.pdf).


Channel pruning method based on BN layers for the  yolov3 and yolov4, we recommond the following repository:

https://github.com/tanluren/yolov3-channel-and-layer-pruning

[coldlarry/YOLOv3-complete-pruning](https://github.com/coldlarry/YOLOv3-complete-pruning)
 
https://github.com/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone

Thanks  for  your contributions.  
