# Face2Webtoon

## Introduction
Face2cartoon is interesting task, but there are few researches applying I2I translation to webtoon. I collected dataset from naver webtoon 연애혁명 and tried to transfer human faces to webtoon domain. 
Link : https://comic.naver.com/webtoon/list.nhn?titleId=570503

## Face Dataset
I used AFAD-Lite dataset from https://github.com/afad-dataset/tarball-lite. 

![image](https://user-images.githubusercontent.com/71681194/104359465-08031b80-5553-11eb-97a3-526a800ee411.png)

## Webtoon Dataset

![data](https://user-images.githubusercontent.com/71681194/104342339-1266ea80-553e-11eb-9e4f-8cd7cbaef418.JPG)


I used anime face detector from https://github.com/nagadomi/lbpcascade_animeface. Since face detector is not that good at detecting the faces from webtoon, I could gather only 1400 webtoon face images.

## Baseline 0(U-GAT-IT)
I used U-GAT-IT official pytorch implementation(https://github.com/znxlwm/UGATIT-pytorch).
U-GAT-IT is GAN for unpaired image to image translation. By using CAM attention module and adaptive layer instance normalization, it performed well on image translation where considerable shape deformation is required, on various hyperparameter settings. Since shape is very different between two domain, I used this model. 

arXiv: https://arxiv.org/abs/1907.10830


![good](https://user-images.githubusercontent.com/71681194/104342049-c61baa80-553d-11eb-9c58-d2d02a5c01aa.jpg)

![gif1](https://user-images.githubusercontent.com/71681194/104342061-c9169b00-553d-11eb-98b1-028c60b513f0.gif)



Some results look pretty nice, but many result have lost attributes while transfering.

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

Even after I collected more data for group 2 and 3, there are severe imbalances between groups. As a result, model failed to translate to few shot groups, for example, group 2 and 3.



## U-GAT-IT + Few Shot Transfer

Few shot transfer : https://arxiv.org/abs/2007.13332

Paper review : https://yun905.tistory.com/48

In this paper, authors successfully transfered the knowledge from group with enough data to few shot groups which have only 10~15 data. First, they trained basic model, and made branches for few shot groups.

### Basic model
For basic model, I trained U-GAT-IT between only group 0.

![basic_model1](https://user-images.githubusercontent.com/71681194/105211139-4326cf80-5b8f-11eb-921d-e0b8761a66ad.jpg)
![basic_model2](https://user-images.githubusercontent.com/71681194/105211143-43bf6600-5b8f-11eb-86d0-8ff37817a003.jpg)

### Baseline 1
For baseline 1, I freeze the bottleneck layers of generator and tried to fine-tune the basic model. I used 38 images(both real/fake) of group 1,2,3, and added 8 images of group 0 to prevent forgetting. I trained for 200k iterations.

![1](https://user-images.githubusercontent.com/71681194/105213333-ed9ff200-5b91-11eb-96c7-a6d46a272d9f.jpg)

Model randomly mapped between groups.

![graph](https://user-images.githubusercontent.com/71681194/105213709-6141ff00-5b92-11eb-986d-0ba83b106839.JPG)

Since the distribution overlap exists, it seems that discriminator overfitting does not occured.
