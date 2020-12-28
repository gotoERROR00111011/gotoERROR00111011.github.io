---
title: "Deep Learning Specialization 1"

categories:
    - Deep Learning

tags :
    - MOOC
    - Coursera
    - Deep Learning

---

## [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)
Coursera Deep Learning Specialization 1  
[Coursera Certificate](https://coursera.org/share/4f5fdf914f54f01a155bec0f4caadbd5)  

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'></script>

## Deep Learning
deep learning 개념은 과거부터 존재했지만, 최근에 많은 주목을 받고있다.  
### Data, Computation, Algorithms  
학습에 사용하는 데이터가 많을수록 성능이 높아지는데, 인터넷, 모바일 등 환경의 변화로 많은 데이터를 얻을 수 있게 된것이 큰 영향을 주었다.  
하드웨어와 알고리즘의 발전 또한 신경망을 더 깊고 효율적으로 만들어 많은 영향을 주었다.  
### Loop (idea -> code -> experiment)  
빨라진 속도로 인해 구현, 실험 기간이 짧아져 생산성이 높아졌다.  
이로 인해 아이디어를 더 신속히 개선할 수 있게 되었다.  

## Logistic Regression
### Binary Classification
logistic regression 은 binary classification 에 사용된다.  
sigmoid 를 사용하여 결과를 0~1 로 변환한다.  

$$x = input$$  
$$W = weight$$  
$$b = bias$$  
  
$$\sigma(z) = \frac{1}{1+e^{-z}}$$  
$$\hat y = \sigma(Wx+b)$$    

### Cost function
loss (error) function : ```predict y```, ```label y``` 의 오차  
cost function : loss function 의 평균  

### Gradient Descent
cost function 이 최소값이 되는 ```W```, ```b``` 를 구하는 방법  
cost function 의 기울기(변화량)가 낮은 쪽으로 최적화 작업을 반복  

### Derivatives
$$\alpha$$ = learning rate  
$$w := w-\alpha\frac{\partial{J(w,b)}}{\partial{w}}$$  
$$b := b-\alpha\frac{\partial{J(w,b)}}{\partial{b}}$$  

### Logistic Regression 에서의 수식
loss function $$L(\hat y, y) = -(y\log(\hat y) + (1-y)\log(1-\hat y))$$  
cost function $$J(w,b) = \frac{1}{m} \sum_{i=0}^{m} L(\hat y^{(i)}, y^{(i)})$$  
derivatives : $$\frac{\partial L(\hat y,y)}{\partial a}=-\frac{y}{a}+\frac{1-y}{1-a}$$  


## Vectorization
deep learning 의 속도 향상을 위해 vectorization 사용  
numpy 의 parallelism 기능을 활용하여 for loop 를 최소한으로 줄인다. (상당한 속도 향상을 확인할 수 있다.)  
SIMD 에 뛰어난 GPU 가 deep learning 유리한 이유  

$$
WX+b=\\
\begin{bmatrix}
\cdots & w_{1} & \cdots\\
\cdots & w_{2} & \cdots\\
\cdots & \vdots & \cdots\\
\cdots & w_{n} & \cdots\\
\end{bmatrix}
\begin{bmatrix}
\vdots & \vdots & \vdots & \vdots\\
x_{1} & x_{2} & \cdots & x_{m}\\
\vdots & \vdots & \vdots & \vdots\\
\end{bmatrix}
+
\begin{bmatrix}
b_{1}\\
b_{2}\\
\cdots\\
b_{n}\\
\end{bmatrix}
\\
=\begin{bmatrix}
w_{1}x_{1}+b_{1}&w_{1}x_{2}+b_{1}&\cdots&w_{1}x_{m}+b_{1}\\
w_{2}x_{1}+b_{2}&w_{2}x_{2}+b_{2}&\cdots&w_{2}x_{m}+b_{2}\\
\vdots & \vdots & \ddots & \vdots&\\
w_{n}x_{1}+b_{n}&w_{n}x_{2}+b_{n}&\cdots&w_{n}x_{m}+b_{n}\\
\end{bmatrix}
$$

## Shallow Neural Network
### Neural Network
input layer  
hidden layer  
output layer  
### Activation Function
$$sigmoid(z) = \frac{1}{1+e^{-z}}$$  
$$\frac{\text{d}}{\text{d}z}sigmoid(z)=sigmoid(z)(1-sigmoid(z))$$  
0과 1 사이의 값  
이진분류 출력층 이외에는 사용하지 않는다.  

$$\tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$$  
$$\frac{\text{d}}{\text{d}z}\tanh(z)=1-\tanh^{2}(z)$$  
sigmoid 보다 우월  
-1과 1 사이의 값  
데이터를 중심에 위치시켜 평균이 0에 가까워지기 때문에 학습이 더 쉽게 이루어진다.  
sigmoid 와 tanh는 z가 매우 크거나 작을 경우 기울기가 매우 작아지는 문제점이 있다.  

ReLU  
$$max(0, z)$$  
$$g'(z)=\begin{cases}0 & \text{if $z<0$}\\1 & \text{if $z>0$}\end{cases}$$  
대부분의 상황에서 ReLU를 사용  

Leaky ReLU  
$$max(0.01*z, z)$$  
$$g'(z)=\begin{cases}0.01 & \text{if $z<0$}\\1 & \text{if $z>0$}\end{cases}$$  

### Non Linear Activation Function
linear activation function(identity activation function) 을 사용할 경우 신경망을 아무리 깊게 구성해도 linear function 일 뿐이기 떄문에 효과가 없다.  
output 으로 실수 값을 원할 경우 출력층에만 linear function 을 사용할수는 있다.  
### Random Initialization
weight를 0으로 모두 동일하게 초기화할 경우 대칭적으로 똑같은 결과만 산출하여 학습이 진행되지 않는다.  


## Deep Neural Network
### Deep Representations
초기 layer 에서는 단순한 작업(이미지-모서리, 소리-높낮이)을 담당하고 깊어 질수록 이전 layer 의 정보를 취합하는 작업을 한다.  

### matrix dimensions
$$
W^{[l]}(n^{[l]},n^{[l-1]})\\
b^{[l]}(n^{[l]},m)
$$

### Forward Propagation
$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}\\
A^{[l]}=g^{[l]}(Z^{[l]})\\
$$

### Chain Rule
$$
\partial A=\frac{\partial L}{\partial A}\\
\partial Z=\frac{\partial L}{\partial Z}={\frac{\partial L}{\partial A}}{\frac{\partial A}{\partial Z}}\\
\partial W=\frac{\partial L}{\partial W}={\frac{\partial L}{\partial A}}{\frac{\partial A}{\partial Z}}{\frac{\partial Z}{\partial W}}\\
$$  

### Back Propagation  
$$
\text{d}Z^{[L]}=A^{[L]}-Y\\
\text{d}W^{[L]}=\frac{1}{m}\text{d}Z^{[L]}A^{[L-1]T}\\
\text{d}b^{[L]}=\frac{1}{m}np.sum(\text{d}Z^{[L]}, axis=1, keepdims=True)\\
\text{d}A^{[L-1]}=W^{[L]T}\text{d}Z^{[L]}\\
\text{d}Z^{[L-1]}=\text{d}A^{[L-1]}*g^{\prime[L-1]}(Z^{[L-1]})\\
\vdots\\
$$  

### Hyperparameters
parameters : ```W```, ```b```  
hyperparameters : ```learning late```, ```#iteration```, ```#hidden layer L```, ```#hidden unit n```, ```choice activation function```, etc...   
  
hyperparameters 가 parameters 를 control  
최적의 hyperparameters 를 찾는 방법은 여러 값들을 직접 시도해보는 것이다.  