# Face2Webtoon

## Introduction
Face2cartoon is interesting task, but there are few researches applying I2I translation to webtoon. I collected dataset from naver webtoon 연애혁명 and tried to transfer human faces to webtoon domain. 
Link : https://comic.naver.com/webtoon/list.nhn?titleId=570503

## AsianFace <-> love revolution
I used AFAD-Lite dataset from https://github.com/afad-dataset/tarball-lite. 

![1](https://user-images.githubusercontent.com/71681194/104017206-0bb53c00-51fb-11eb-8e3a-2fbdcb93f1d8.jpg)

## Webtoon Dataset

![data](https://user-images.githubusercontent.com/71681194/104342339-1266ea80-553e-11eb-9e4f-8cd7cbaef418.JPG)


I used anime face detector from https://github.com/nagadomi/lbpcascade_animeface. Since face detector is not that good at detecting the faces from webtoon, I could gather only 1400 webtoon face images.

## U-GAT-IT(Baseline)
I used U-GAT-IT official pytorch implementation(https://github.com/znxlwm/UGATIT-pytorch).
U-GAT-IT is GAN for unpaired image to image translation. By using CAM attention module and adaptive layer instance normalization, it performed well on image translation where considerable shape deformation is required, on various hyperparameter settings. Since shape is very different between two domain, I used this model. 

arXiv: https://arxiv.org/abs/1907.10830


### Missing of Attributes

![good](https://user-images.githubusercontent.com/71681194/104342049-c61baa80-553d-11eb-9c58-d2d02a5c01aa.jpg)

![gif1](https://user-images.githubusercontent.com/71681194/104342061-c9169b00-553d-11eb-98b1-028c60b513f0.gif)

Some results look pretty nice, but many result have lost attributes while transfering.

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

Even after I collected data for group 2 and 3, webtoon dataset has severe imbalances between groups. As a result, source images were translated to dominant group(group 0) so that attributes like gender, glasses were lost.

FID score : 150.23

## U-GAT-IT + Few Shot Transfer
