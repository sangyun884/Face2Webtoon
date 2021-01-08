# WebtoonFaces

## Introduction
I tried image to image translation between human face to webtoon, and vice versa. I used dataset from naver webtoon love revolutioin.
Link : https://comic.naver.com/webtoon/list.nhn?titleId=570503

## Webtoon Dataset
I used anime face detector from https://github.com/nagadomi/lbpcascade_animeface. Since face detector can't extract the faces from webtoon, I could gather 1604 webtoon face images.

## U-GAT-IT
I modified U-GAT-IT official pytorch implementation(https://github.com/znxlwm/UGATIT-pytorch)
U-GAT-IT is GAN for unpaired image to image translation. By using CAM attention module and adaptive layer instance normalization, it performed well on image translation where considerable shape deformation is required, on various hyperparameter settings. Since texture is very different between two domain, I used it. 

arXiv: https://arxiv.org/abs/1907.10830

## AsianFace <-> love revolution
I used AFAD-Lite dataset from https://github.com/afad-dataset/tarball-lite. 

