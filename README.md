# Face2webtoon

![merge_from_ofoct (2)](https://user-images.githubusercontent.com/71681194/108319761-35538100-7205-11eb-80fe-aa4ba1400d80.jpg)

![merge_from_ofoct (1)](https://user-images.githubusercontent.com/71681194/108319763-3684ae00-7205-11eb-99ff-f3289ee99498.jpg)


## Introduction
Image to image (I2I) translation is an extremely challenging task, especially when data is insufficient. This work aims at transferring human faces to the webtoon domain with limited data.


## Webtoon Dataset

![data](https://user-images.githubusercontent.com/71681194/104342339-1266ea80-553e-11eb-9e4f-8cd7cbaef418.JPG)


I collected dataset from naver webtoon [연애혁명](https://comic.naver.com/webtoon/list.nhn?titleId=570503) using [anime face detector](https://github.com/nagadomi/lbpcascade_animeface). Since face detector is not that good at detecting the faces from webtoon, I could gather only 1400 webtoon face images.

## Baseline 0(U-GAT-IT)
I used [U-GAT-IT official pytorch implementation](https://github.com/znxlwm/UGATIT-pytorch).
[U-GAT-IT](https://arxiv.org/abs/1907.10830) is GAN for an unpaired I2I translation. Using CAM attention module and adaptive layer instance normalization, it performed well on I2I translation where considerable shape deformation is required.

For face data, i used AFAD-Lite dataset from https://github.com/afad-dataset/tarball-lite. 




![good](https://user-images.githubusercontent.com/71681194/104342049-c61baa80-553d-11eb-9c58-d2d02a5c01aa.jpg)

![gif1](https://user-images.githubusercontent.com/71681194/104342061-c9169b00-553d-11eb-98b1-028c60b513f0.gif)



Some results look pretty nice, but some attributes were lost during the domain transfer.

### Missing of Attributes

#### Gender

![gender](https://user-images.githubusercontent.com/71681194/104342136-db90d480-553d-11eb-9f47-939e1f7e1b0d.jpg)

Gender information was lost.

#### Glasses

![glasses](https://user-images.githubusercontent.com/71681194/104342163-e0ee1f00-553d-11eb-9aec-6c7c7aae64b1.jpg)

A model failed to generate glasses in the webtoon faces.

### Result Analysis

To analysis the result, I seperated webtoon dataset to 5 different groups.

|group number|group name|number of data|
|---|---|---|
|0|woman_no_glasses|1050|
|1|man_no_glasses|249|
|2|man_glasses|17->49|
|3|woman_glasses|15->38|

Even after I collected more data for groups 2 and 3, there are severe imbalances between groups. As a result, the model failed to translate to few shot groups, for example, groups 2 and 3.



## U-GAT-IT + Few Shot Transfer

Few shot transfer : https://arxiv.org/abs/2007.13332

Paper review : https://yun905.tistory.com/48

In this paper, the authors successfully transferred the knowledge from the group with enough data to few shot groups.

### Basic model
The basic model was trained for group 0 only.

![basic_model1](https://user-images.githubusercontent.com/71681194/105211139-4326cf80-5b8f-11eb-921d-e0b8761a66ad.jpg)
![basic_model2](https://user-images.githubusercontent.com/71681194/105211143-43bf6600-5b8f-11eb-86d0-8ff37817a003.jpg)

### Baseline 1 (simple fine-tuning)
For baseline 1, I froze the bottleneck layers of the generator and tried to fine-tune the basic model. I used 38 images(both real/fake) of groups 1,2 and 3. Additionally, I also used 8 images of group 0 to prevent forgetting. The model was trained for 200k iterations.

![1](https://user-images.githubusercontent.com/71681194/105213333-ed9ff200-5b91-11eb-96c7-a6d46a272d9f.jpg)

Model randomly mapped between groups.

### Baseline 2 (group classification loss + selective backprop)

![0](https://user-images.githubusercontent.com/71681194/106051720-2c9eec00-612c-11eb-8c73-7c8deba76e1d.jpg)

I attached additional group classifier to discriminator and added group classification loss according to [original paper](https://arxiv.org/abs/2007.13332). Images of groups 0,1,2,3 were fed sequentially, and bottleneck layers of the generator were updated for group 0 only.

Since FID has a significant amount of bias when training with a limited data, I used [KID](https://github.com/abdulfatir/gan-metrics-pytorch)
|KID*1000|
|---|
|25.95|

## U-GAT-IT + group classification loss + adaptive discriminator augmentation
[ADA](https://arxiv.org/abs/2006.06676) is a useful data augmentation method for training GAN with limited data. Although the authors only considered unconditional GANs, I applied ADA to U-GAT-IT. Augmentation was applied to both discriminators. I assumed that preventing the discriminator of the face domain from overfitting would improve the performance of the face generator. Therefore the cycle consistency loss would be more meaningful. Only pixel blitting and geometric transformation was implemented. The effects of other augmentation methods are minimal, according to the paper.

To achieve better result, I changed face dataset to more diverse one([CelebA](https://www.kaggle.com/jessicali9530/celeba-dataset)).


![merge_from_ofoct (2)](https://user-images.githubusercontent.com/71681194/108319761-35538100-7205-11eb-80fe-aa4ba1400d80.jpg)

![merge_from_ofoct (1)](https://user-images.githubusercontent.com/71681194/108319763-3684ae00-7205-11eb-99ff-f3289ee99498.jpg)



![image](https://user-images.githubusercontent.com/71681194/108319007-199bab00-7204-11eb-9cc9-08f3d199816d.png)

ADA makes training longer. I trained for 8 days with a single 2070 SUPER but did not fully converge.

|KID*1000|
|---|
|12.14|

## Checkpoint file

https://drive.google.com/file/d/1uS5-3iIbYeh_aau0ydNkTO5aRhJWw0jX/view?usp=sharing


## Start training

python main.py --dataset dataset_name --useADA True --group 0,1,2,3 --use_grouploss True --neptune False

If --neptune is True, the experiment is transmitted to neptune ai, which is experiment management tool. You must set your API token. --group 0,1,3 make group 2 out of training.
