# Facetoanimals
Source : https://github.com/AUTOMATIC1111/stable-diffusion-webui

**이 프로젝트는 MS AI School 2기 팀프로젝트로 활용할 계획이며, 단순 교육 목적임을 밝힙니다.**

**this project is going to be used as the team project for the 2nd term of MS AI School, and it is expressly for educational purposes.**

가) 프로젝트 계획서
사람의 얼굴을 강아지 얼굴로 바꾸는 프로젝트입니다. 

OS : Windows\
GPU : 1 x Tesla V100 16GB, 1 x T4 16GB\
적용할 관련 모델 : Stable-Diffusion

<br>

## 개요

### 배경
### 목적
### 목표
### 기대효과


## 팀원 소개


## Method

## Results

## Conclusion

<br>
<br>

이하는 저희가 하루에 한 번씩 진행 경과를 위해 작성하는 공간입니다. \
<br>
<br>

### <08.21>
- 사람 얼굴로 닮은 동물 이미지를 생성하는 것이 주요 목표
- 사람 얼굴의 특징점 찾기, 동물 얼굴의 특징점 찾기를 먼저 진행 해서 데이터를 모아야 할 것이라고 생각함
- 초반에는 각자 모델 하나씩 잡아서 특징점 찾기를 돌려보고 성능비교
- 평가 코드까지 작성해보고 각 모델의 장단점 특징 이런걸 정리하고 그 중 하나를 픽 하는게 좋을 것
- T4로 돌려보고 속도가 충분치 않다면 V100
- yolo8, resnet101, facenet 등등 각자 모델 하나씩 잡아보고 돌려보기
- Flask 나 Django 를 사용해서 웹페이지나 어플리케이션 만드는거

*일단 yolov8 모델로 돌려보면서 라벨링 검증*

### <08.22>

<img src=.\outputs\readme\Golden_Retriever_class.jpg>\
0: 얼굴

1: 오른쪽 눈

2: 코

3: 왼쪽

- 이런식으로 특징점 잡아보자

유사도 거리

- 지금 개 사진이 기울어져있는 것도 있고, 정면인것도 있는데
- 이런걸 테이블 처리해서 enable?
- 모델 찾아보고
- 사람 사진을 모델이 넣어보거나
- 동물 사진을 사람 모델링에 훈련시켜보기

### <08.23>

model : 사람 keypoints 모델

input : 강아지 사진

output : 좌표를 어느 정도 잡는다.

이후
Diffusion

**Reference** \
<br>
[YOLOv8-face-landmarks-opencv-dnn](https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn)

[YOLOv8-face](https://github.com/akanametov/yolov8-face)

[Image to Image : Huggingface-diffusers](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)

[Style GAN2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

[img2text interrogator : huggingface_CLIP_interrogator](https://huggingface.co/spaces/pharmapsychotic/CLIP-Interrogator)

- 디퓨전 모델을 활용하여 강아지 얼굴로 변화

[Palette-Image-to-Image](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
### <08.24>

### <08.25>


프로젝트 5일차, 모델 선정 3일차


**face, Golden Retriever**\
<img src=.\outputs\txt2img-images\2023-08-25\face_Golden_Retriever.png>