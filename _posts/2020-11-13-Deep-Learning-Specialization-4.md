---
title: "Deep Learning Specialization 4"

categories:
    - Deep Learning

tags :
    - MOOC
    - Coursera
    - Deep Learning

---

## [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)
Coursera Deep Learning Specialization 4   
[Coursera Certificate](https://coursera.org/share/219496f25bdc373b59d1f56f351d031a)  

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'></script>



## Convolutional Neural Networks
### Filter (Kernel)
이미지를 fully connected network 로 처리하면 pixel 수에 비례하여 weights 가 커진다. filter 를 weights 로 사용하면 이러한 문제점을 해결할수있다.  
convolution 에서는 sobel, scharr, etc... 등 여러 filter 를 사용해 이미지의 feature 를 찾는다. input data 의 feature 에 맞추어 filter 를 학습하는것이 convolution neural network 이다.  
parameters(filter) 의 크기도 작기 때문에 overfitting 확률도 낮아진다.  
### Padding
이미지의 가장자리를 추가 데이터로 감싸는 방법  
convolution 에 의해 이미지의 가장자리 데이터가 삭제, 크기가 줄어드는 것을 해결할때 사용한다.  
padding 을 사용하지 않는 valid padding, input-output 크기를 동일하게 유지하는 same padding 이 있다.  
### Stride
convolution 실행 시, filter 의 이동 수치  
### Pooling
feature 를 감지하고, size 를 줄일수있다.  
max pooling, average pooling 등이 있으며 주로 max pooling 을 사용한다.  
### Output Size
input size, filter size, padding, stride 으로 간단하게 output size 를 계산해볼수있다.  
$$ \lfloor \frac{n+2p-f}{s}+1 \rfloor \\$$  
### Layer
weights 를 기준으로 1 layer 로 구분한다. pooling 은 weights 가 없으므로 convolution 과 함께 1 layer 로 구분한다.  



## Classic Network
(논문은 google scholar 에서 검색)  
### LeNet-5
> gradient based learning applied to document recognition  

### AlexNet
> Imagenet classification with deep convolutional neural networks

### VGG-16
> Very deep convolutional networks for large-scale image recognition

### ResNet
> Deep residual learning for image recognition

#### Residual Block
activation 을 다음 단계의 layer 의 activation function 로 shortcut, skip 하는 방법  
$$a^{[l+2]}=g(z^{[l+2]}+a^{[l]})\\$$  
  
이론적으로 network 가 깊어지면 error 는 낮아져야 하지만, 현실적으로 너무 깊어지면 error 가 높아진다. 이러한 문제점을 해결하기위해 사용한다.  
  
ReLU 를 사용하면 activation>=0 이고, 0일 경우에도 skip 한 layer 는 identity function 이 되기 때문에, 적어도 성능 저하는 발생하지 않는다.  
  
다른 layer 에서 사용하기 위해서는 shape 가 같아야하기 때문에 $$W_{s}$$ 를 사용해 조정한다. ($$W_{s}$$는 parameters or 고정)  
### 1X1 Convolution
> Network in network
  
1 X 1 X channel 구조  
channel 수 조정이 가능하다.  
  
input(28, 28, 192), filter(5, 5, 32), convolution 연산량 28x28x32 x 5x5x192 = 120,422,400  
  
input(28, 28, 192), filter(1, 1, 16), convolution 연산량 28x28x16 x 1x1x192 = 2,408,448  
input(28, 28, 16), filter(5, 5, 32), convolution 연산량 28x28x32 x 5x5x16 = 10,035,200  
  
중간에 1x1 convolution 을 사용하여 channel 을 줄여주면 연산량을 줄일수 있다. (120,422,400 vs 12,443,648)  
### Inception Network
> Going deeper with convolutions  
  
1X1, 3X3, 5X5, max pooling 을 합치는 구조.  



## Technic, Tools, TIP
### Open Source
새로운 network, hyperparameters 를 찾는것보다 이미 검증된 논문을 사용하는게 효율적이다. 하지만 논문을 구현하는것은 어렵다. (dataset, training, hyperparameters tuning 등)  
github 에 공개되어있는 network, pretrained parameters 등을 활용하면 시간, 성능 측면에서 큰 도움이 된다.  
### Transfer Learning
pretrained parameters 를 다운로드하여 transfer learning 에 사용할수있다.  
자신의 data set 의 크기에 따라서 다음과 같은 transfer learning 이 가능하다.  
1. 전체 layer 들을 고정시키고 softmax 단계만 학습
1. 초기 layer 만 고정시키고 후기 layer 학습
1. pretrained parameters 를 initialization 으로 사용하여 전체 학습

이외에도 중간 layer 까지만 그대로 사용하고 layer 를 변형, 추가시켜 사용할수도 있다.  
transfer learning 에서 학습하지 않는 (고정시킨) layer 까지의 출력 결과를 미리 계산하여 저장해두고 사용하면 중간까지의 연산을 반복하지 않아도 된다.  
### Data Augmentation
여러가지 방법이 있다.  
- Mirroring
- Random Cropping
- Rotation
- Shearing
- Local Warping
- Color Shifting

일반적으로 학습과정에서 적용한다.  
### State of Computer Vision
#### Ensembling
다수의 network 의 output 평균 사용하는 방법  
1~2% 정도 정확도 향상이 가능하지만 속도가 느려져서 제품에는 부적합하다.  
#### Multi-Crop
하나의 이미지를 여러번 crop 하고, 하나의 network 로 각 crop 에 대한 output 을 평균내는 방법  


## Detection algorithms
### Object Localization
input image 에서 물체가 있는 영역을 인식하는 방법  
output(p(y or n), x, y, w, h, class1,2,3...)  
### Landmark Detection
좌표(x,y)를 인식하는 방법  
ex) 얼굴(손, 손가락 마디, 팔꿈치, 어깨...)  
### Object Detection
#### Sliding Windows
이미지 전체 영역을 일정 간격을 두고 이동하며 적용시키는 방법  
각 window 마다 convolution neural network 를 사용하면 window 가 겹치는 영역을 중복으로 convolution 하게되고 속도가 느려진다.  
#### Convolutional  
> Overfeat: Integrated recognition, localization and detection using convolutional networks  

sliding windows 의 속도 문제는 convolutional 하게 구현하여 해결할수있다.  
1. 전체 이미지를 input 으로 convolution neural network 를 진행한다.
1. fully connected layer 를 (1, 1, channel) 형태로 한다.
1. 마지막 출력 channel 은 class 수를 가지도록 설정한다.
1. softmax 를 적용한다.

전체 이미지 단위로 convolution 하기 때문에 sliding windows 의 window 사이의 겹치는 영역의 중복 convolution 으로 인한 속도저하가 없어진다. 또한 이미지 크기에 제한받지 않는다. (이미지 크기가 작은 경우는 제외)  
bounding box 의 위치가 정확도가 낮다는 단점이 있다.  
### YOLO
> You only look once: Unified, real-time object detection  

localization 을 convolutional 하게 구현하여 object detection 하는 algorithm  
classification 대신 localization 을 사용하여 sliding windows 의 문제점인 bounding box 의 부정확함을 해결했다.  
#### Bounding Box Predictions
output channel 의 x, y, w, h 는 해당 cell 기준이다.  
중심 좌표 x, y 는 grid cell 내부에 존재해야하지만, 크기 w, h 의 범위는 grid cell 보다 클수도 있으며 cell 보다 큰 object 도 detection 할수있다.  
#### Intersection Over Union
box(ground truth, bounding box) 가 겹치는 비율을 계산하는 방법 (겹치는영역/전체영역)  
$$IoU=\frac{(ground\ truth \ \cap\ bounding\ box)}{(ground\ truth \ \cup\ bounding\ box)}\\$$  
#### Non-max Suppression
같은 class 이고, 겹치는 bounding box 를 제거하는 방법  
예측치가 가장 높은 bounding box 를 선택하고, 이 bounding box 와 일정 이상의 IoU 를 가지는 bounding box 를 제거하고, 이 과정을 반복한다.  
#### Anchor Boxes
예측할 object 의 크기, 비율을 가진 anchor box 들을 미리 정의해놓은것  
cell 마다 각각의 anchor box 와 높은 IoU 를 갖는 object 를 찾는다. cell 마다 anchor box 의 수만큼 찾아서 1개 밖에 못찾던 문제를 해결한다.  
anchor box 의 크기는 K-means clustering 을 사용하는것이 정확도가 높다.  
output(grid w, grid h, (anchor-box1(p(y or n), x, y, w, h, class1,2,3...), anchor-box2...))  
#### Region Proposals (segmentation)
object 가 있을것이라 예상되는 후보 region 을 예측하고, 후보 region 을 classification 하는 방법  
- R-CNN
- Fast R-CNN
- Faster R-CNN

YOLO 보다 느리다.  



## Face Recognition
### One Shot Learning
한개의 데이터만으로 학습하는것.  
softmax 로는 잘 동작하지 않는다.(데이터 부족 + 대상이 추가될 때마다 매번 다시 학습해야함)  
### Similarity Function
두 대상의 유사도를 측정하는 function  
### Siamese Network
> Deepface: Closing the gap to human-level performance in face verification  
  
classification 의 최종 encoding 을 fully connected layer 로 하고, 비교할 두 대상의 encoding 을 비교한다.
$$ distance={\left\lVert f(x^{(i)})-f(x^{(j)}) \right\rVert}^{2} \\$$  
### Triplet Loss
> Facenet: A unified embedding for face recognition and clustering  
  
$$
A(Anchor),\ P(Positive),\ N(Negative)\\
{\left\lVert f(A)-f(P) \right\rVert}^{2} \leq 
{\left\lVert f(A)-f(N) \right\rVert}^{2}\\
{\left\lVert f(A)-f(P) \right\rVert}^{2} - 
{\left\lVert f(A)-f(N) \right\rVert}^{2} \leq 0\\
$$  

서로 다른 input 에 대하여 $$f(x)$$의 encoding 은 같아서는 안된다.  
또한 $$f(x)$$가 0만을 encoding 한다면 $$0-0 \leq 0$$ 가 성립하게 되기 때문에 margin $$\alpha$$를 더해준다.  
$$
{\left\lVert f(Anchor)-f(Positive) \right\rVert}^{2} - 
{\left\lVert f(Anchor)-f(Negative) \right\rVert}^{2} + \alpha \leq 0\\
$$  
loss 는 위의 조건을 만족하지 못한 경우의 수치를 사용한다.  
$$
L(A,P,N)=max({\left\lVert f(Anchor)-f(Positive) \right\rVert}^{2} - 
{\left\lVert f(Anchor)-f(Negative) \right\rVert}^{2} + \alpha,\ 0)\\
$$  

triplet A, P, N 을 random 하게 선택한다면 $$distance(A, N)$$은 큰 값이 될것이고, $$distance(A,P)+\alpha \leq distance(A,N)$$을 쉽게 충족시켜서 network 는 제대로 학습하지 못한다.  
$$distance(A, P)$$와 $$distance(A, N)$$이 비슷한 값을 가지도록 triplet A, P, N 을 선택해야 제대로된 학습이 이루어진다.  
### Face Verification and Binary Classification
triplet 대신 두 이미지의 encoding 결과를 binary classification 의 input 으로 사용하여 학습하는 방법도 있다.  
### TIP
database 의 얼굴 사진들의 encoding 결과를 미리 저장해두고 재사용하여 효율을 높일수있다.  



## Neural Style Transfer
### What are deep ConvNets learning
> Visualizing and understanding convolutional neural networks  

shallow layer 에서는 texture 를, deep layer 에서는 object 를 인식한다.  
### Cost Function
> A neural algorithm of artistic style  
  
generated 이미지를 평가하는 $$J(G)$$ 는 content 이미지와 generated 이미지가 얼마나 비슷한지 측정하는 $$J_{content}(C, G)$$, style 이미지와 generated 이미지가 얼마나 비슷한지 측정하는 $$J_{style}(S, G)$$ 의 합이다.  
$$J(G)=\alpha J_{content}(C,G)+\beta J_{style}(S,G)\\$$  
$$J$$는 한개의 layer 에서 수행한다. 이 layer 가 얕으면 input 과 큰 차이가 없고, 너무 깊으면 classification 에 가까워진다. 따라서 너무 얕지도, 깊지도 않은 적당한 l layer 를 선택해야한다.  
#### Content Cost Function
두 이미지의 유사도를 평가하는 $$J_{content}(C,G)$$ 를 l layer 에 사용한다.  
$$J_{content}(C,G)=\frac{1}{2}{\left\lVert a^{[l](C)}-a^{[l](G)} \right\rVert}^{2}\\$$  
#### Style Cost Function
layer 에서 찾아낸 texture(channel) 사이의 상관관계로 각 feature 가 얼마나 같이 발생하는지 판단할수있다.  
ex) 파란색 feature, 수직 feature 두개의 상관관계가 크면 파란색 수직 feature 가 존재할것이고 상관관계가 작으면 반대일것이다.  

texture(channel) 사이의 상관관계를 측정하는 
$$
G^{[l](G)}_{kk\prime}=\sum_{i=1}^{n_H}\sum_{j=1}^{n_W}a^{[l](G)}_{i,j,k}a^{[l](G)}_{i,j,k^\prime}
$$
을 l layer 전체에 적용시킨다.  
$$
J^{[l]}_{style}(S,G)=\frac{1}{2n^{[l]}_Hn^{[l]}_Wn^{[l]}_C}\sum_k\sum_{k^\prime}(G^{[l](S)}_{kk^\prime}-G^{[l](G)}_{kk^\prime})^{2}\\
$$  
다음처럼 다른 layer 에서도 $$J_{style}$$ 을 사용하면 더 좋은 결과를 얻을수있다. (저수준, 고수준의 feature)  
$$J_{style}(S,G)=\sum_{l}\lambda^{[l]}J_{style}^{[l]}(S,G)\\$$  



## 1D and 3D Generalizations
1D, 3D 또한 각 차원에 맞는 filter 로 convolution 가능하다.  
