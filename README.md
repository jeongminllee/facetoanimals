# Facetoanimals
Source : https://github.com/AUTOMATIC1111/stable-diffusion-webui

**이 프로젝트는 MS AI School 2기 팀프로젝트로 활용할 계획이며, 단순 교육 목적임을 밝힙니다.**

**this project is going to be used as the team project for the 2nd term of MS AI School, and it is expressly for educational purposes.**

가) 프로젝트 계획서
사람의 얼굴을 강아지 얼굴로 바꾸는 프로젝트입니다. 

python 3.10.6
OS : Windows\
GPU : 1 x Tesla V100 16GB, 1 x T4 16GB\
적용할 관련 모델 : Stable-Diffusion


(env) ~~/stable-diffusion-webui>webui-user.bat
<br>

## 개요

### 배경
### 목적
### 목표
### 기대효과


## 팀원 소개

이정민 : 

이남열 : 

이성범 : 

이주형 : 

이훈민 : 


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

![Golden_Retriever_class](https://github.com/jeongminllee/facetoanimals/assets/129810866/7dd44762-84bc-4205-af6a-55d4ef53a616)\
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

[Palette: Image-to-Image Diffusion Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)

[Review: Palette: Image-to-Image Diffusion Models](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/palette/)

![스크린샷 2023-08-25 162252](https://github.com/jeongminllee/facetoanimals/assets/129810866/92f0ca14-47ee-4d21-85b5-816f83551897)

특징점을 인식한 상태로 조건부 학습

특징점을 가진 강아지로 변형

- 자원이 모자랄 것 같아 강아지의 품종을 하나로 고정.
- 사람의 특징점도 일단은 한사람의 특징점으로 고정(잘 되면 여러 특징점을 학습)

### Stable Diffusion

[개발자로서 StableDiffusion 사용을 위해 알아두면 좋은 내용들](https://haandol.github.io/2023/07/16/stable-diffusion-for-developers.html#fn:2)


### <08.24>

[Training Stable Diffusion with Dreambooth using 🧨 Diffusers](https://huggingface.co/blog/dreambooth)

[diffusers](https://github.com/huggingface/diffusers)

![00126-3947778887](https://github.com/jeongminllee/facetoanimals/assets/129810866/3ff3f53b-7ede-4aa1-b910-db2e177c1aa0)

![dafdafdafdaf](https://github.com/jeongminllee/facetoanimals/assets/129810866/06f4fe83-20da-4ab8-9344-5edc996c70a6)

![MicrosoftTeams-image (5)](https://github.com/jeongminllee/facetoanimals/assets/129810866/4a8e4a59-e7f5-4326-8bd8-d1075286c354)


생각보다 성능이 좋았지만, 우리가 원하는 성능에는 못 미치는 성능이기 때문에 보완할 부분을 찾아보고자 함.

### <08.25>


프로젝트 5일차, 모델 선정 3일차

```
사람 얼굴을 기반으로(특징을잡아) 개의 얼굴을 생성하는 문제
-> 스테이블디퓨전 컨트롤넷으로 완벽하게 해결가능(조건부생성단을 우리가 안만들어도됨..)

현상황 -> 기본 모델+컨트롤넷의 테스트로 어느부분을 파인튜닝해야 할지 측정중

if) 테스트결과

A. 부족한 부분이 발견 될 시 파인튜닝을한다.

case1. -> 개의 얼굴 형상이 구현이 잘 안 될시
dog's face 태그에 강아지 얼굴 사진을 학습시켜 스테이블디퓨전을 파인튜닝한다
-> 잘 될 경우 case2로 넘어간다

case2. -> 생성되는 개의 얼굴의 품종이 계속 다를시
사람 얼굴에서 나오는 특정 특징들(좌표를 제외한 쳐진눈,귀의모양,머리색등)을 묶어
해당 특징이 나오는 개의 품종이 도출되게
언어모델을 파인튜닝한다

B. 부족한 부분이 없을 시...
case1. 스테이블 디퓨전 모델이 기능이 너무 많고 확장성도 뛰어나며 가볍고 엄청난모델이기에..
코드를 뜯고 자세하게 분석하며 만들고 끝을 낸다..
case2. 아예 다른주제로 넘어가거나 특정 스타일을 학습시키는 새로운 주제를 한다..
```

[How does Stable Diffusion work?](https://stable-diffusion-art.com/how-stable-diffusion-work/)

</br>

![MicrosoftTeams-image (7)](https://github.com/jeongminllee/facetoanimals/assets/129810866/039048b5-598d-4b5b-bc89-b25c695d978b)


**face, Golden Retriever**\
![스크린샷 2023-08-25 144833](https://github.com/jeongminllee/facetoanimals/assets/129810866/3201a4f7-98b5-41fc-a440-c31a14edbc7b)


![MicrosoftTeams-image (9)](https://github.com/jeongminllee/facetoanimals/assets/129810866/bbb9a24c-5676-4085-bd24-defca6e3a0eb)

![MicrosoftTeams-image (8)](https://github.com/jeongminllee/facetoanimals/assets/129810866/834a1687-6514-4a7e-8849-78c04a64d8b9)

```
dog,close up, face portrait\
Steps: 20, Sampler: DPM++ 2S a, CFG scale: 7, Seed: 1934093539, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, ControlNet 0: "Module: openpose_full, Model: None, Weight: 1, Resize Mode: Crop and Resize, Low Vram: True, Processor Res: 512, Guidance Start: 0, Guidance End: 1, Pixel Perfect: False, Control Mode: ControlNet is more important", Version: v1.5.2
```

![eeeeff](https://github.com/jeongminllee/facetoanimals/assets/129810866/3244de97-2ef7-420e-8a15-2adb7e50f589)


```
dog,close up, (face portrait 1.1), front face
Steps: 20, Sampler: DPM++ 2S a, CFG scale: 7, Seed: 999672668, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, ControlNet 0: "Module: mediapipe_face, Model: None, Weight: 1.25, Resize Mode: Crop and Resize, Low Vram: True, Processor Res: 512, Threshold A: 1, Threshold B: 0.5, Guidance Start: 0, Guidance End: 1, Pixel Perfect: False, Control Mode: ControlNet is more important", Version: v1.5.2

소요 시간: 5.9 sec.
A: 3.11 GB, R: 3.72 GB, Sys: 4.7/15.8457 GB (29.8%)
```