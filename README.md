# WebtoonFaces

## Introduction
I tried image to image translation between human face to webtoon, and vice versa. I used dataset from naver webtoon love revolutioin.
Link : https://comic.naver.com/webtoon/list.nhn?titleId=570503

## Webtoon Dataset

![data](https://user-images.githubusercontent.com/71681194/104017792-16240580-51fc-11eb-8382-2e97c9205fe5.JPG)


I used anime face detector from https://github.com/nagadomi/lbpcascade_animeface. Since face detector is not that good at detecting the faces from webtoon, I could gather only 1604 webtoon face images.

## U-GAT-IT
I used U-GAT-IT official pytorch implementation(https://github.com/znxlwm/UGATIT-pytorch).
U-GAT-IT is GAN for unpaired image to image translation. By using CAM attention module and adaptive layer instance normalization, it performed well on image translation where considerable shape deformation is required, on various hyperparameter settings. Since texture is very different between two domain, I used this model. 

arXiv: https://arxiv.org/abs/1907.10830

## AsianFace <-> love revolution
I used AFAD-Lite dataset from https://github.com/afad-dataset/tarball-lite. 

![1](https://user-images.githubusercontent.com/71681194/104017206-0bb53c00-51fb-11eb-8e3a-2fbdcb93f1d8.jpg)

### Missing of Attributes

Some results looks pretty nice, but many result have missing attributes while transfering.

#### Gender

![attribute1](https://user-images.githubusercontent.com/71681194/104017342-4cad5080-51fb-11eb-8a5f-a1c443133e1c.jpg)

Gender information was reversed.

#### Glasses

![attribute2](https://user-images.githubusercontent.com/71681194/104017721-fa206400-51fb-11eb-9456-6b1a7ec4e975.jpg)

A model failed to generate glasses in the webtoon faces. I guess the reason is that only few characters are wearing glasses in webtoon data while many people are wearing glasses in face data. 

#### Hands

![attribute3](https://user-images.githubusercontent.com/71681194/104029423-4fb13c80-520d-11eb-8ec7-25794270ee40.jpg)

Since there are only few images of hand appearing in webtoon dataset, model compensated that part of image with background.

Given results above, I think we can't transfer the attributes which doesn't exist in target domain, because the generated image with attributes from source domain that doesn't exist in target domain doesn't seem plausible to discriminator. Webtoon faces wearing glasses are needed to transfer the glasses of real image.

### Failure cases : face -> webtoon

![fate2webtoon](https://user-images.githubusercontent.com/71681194/104030183-58564280-520e-11eb-804d-4f9f152042b4.jpg)

Image translation from webtoon to real face is much difficult task, because destination domain has more information than source domain. The results are not good.
