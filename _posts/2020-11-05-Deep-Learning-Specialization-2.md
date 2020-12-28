---
title: "Deep Learning Specialization 2"

categories:
    - Deep Learning

tags :
    - MOOC
    - Coursera
    - Deep Learning

---

## [Improving Deep Neural Networks](https://www.coursera.org/learn/deep-neural-network)
Coursera Deep Learning Specialization 2  
[Coursera Certificate](https://coursera.org/share/d09ff66c2a2acd453db13193a521ff4e)  

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'></script>

## Data Sets
data set은 train set, dev set, test set 등으로 나누어 사용한다. (dev set 은 생략하기도 한다.)  
주로 70:30, 60:20:20 비율로 사용한다.  
하지만 전체 data 가 충분히 많다면 성능 평가에 필요할 정도의 크기로만 test set 을 사용해도 된다.  
  
균일하지 않은 유형의 data 를 dev set, test set 에 맞추는것도 중요하다.  
ex) 이미지(화질, 품질)  

## Bias / Variance
학습 성능이 잘 나오지 않는 underfitting(high bias) 상태  
train set 에 과도하게 학습된 overfitting(high variance) 상태  
  
1. high bias ?
    1. underfitting, training set performance
    1. bigger network, time longer, (NN architecture search)
1. high variance ?
    1. overfitting, test set performance
    1. more data, regularization, (NN architecture serach)


## Regularization
regularization 을 사용하면 overfitting 를 방지하는데 도움을 준다.  

### L2 regularization  
(hyperparameter : $$\lambda$$)  

$$
J(W, b)=\frac{1}{m}\sum_{i=1}^mL(\hat y^{i}, y^{i})+\sum_{l=1}^{L}\frac{\lambda}{2m}{\left\lVert w^{[l]} \right\rVert}_{2}\\
$$  

$$
\frac{\lambda}{2m}{\left\lVert w^{[l]} \right\rVert}_{2}=\frac{\lambda}{2m}\sum_{i=1}^{n^{[l]}}\sum_{j=1}^{n^{[l-1]}}{(w_{i,j}^{[l]})}^2\\
$$  

#### weight decay
다음의 특성 $$(1-\alpha\frac{\lambda}{m})w$$ 으로 인해 weight decay 라고 불리기도 한다.  
$$
{\partial w}=\frac{\partial L}{\partial w}+\frac{\lambda}{m}w\\
w:=w-\alpha{\partial w}\\
w:=(1-\alpha\frac{\lambda}{m})w-\alpha{\partial w}\\
$$  
  
$$\lambda\uparrow$$이면 $$W\downarrow,(Z=Wx+b)\downarrow$$인데,  
$$\text{sigmoid}$$, $$\tanh$$ 같은 activation function 의 경우 $$Z$$의 값이 작을수록 선형 그래프에 가까워진다.  
이로인해 simple 하게 되어 overfitting 을 줄여준다.  

#### L1 regularization  
$$
\frac{\lambda}{2m}{\left\lVert w \right\rVert}_{1}=\frac{\lambda}{2m}\sum_{j=1}^{l}{\lvert w_{j} \rvert}\\
$$  

### Dropout (Inverted Dropout) regularization
(hyperparameter : ```keep_prob```)  
일정 비율의 hidden unit 을 random 하게 비활성화 시키는 방법  
0~1(0%~100%)로 활성화 비율을 지정하고, 비활성화로 낮아진 출력을 ```A/=keep_prob```으로 높여준다.  
test 에서는 임의의 경우를 사용하는 것은 부적절하기 때문에 사용하지 않는다.  

### Other regularization
Data Augumentation (사진 회전, 대칭)  
Early Stopping (test 의 error 가 상승하는 지점에서 학습 정지)  

## Optimization
### Nomalizing
nomalizing 을 하면 cost function 이 구형에 가까운 형태(치우치지 않은 형태)가 되어 경사하강법 과정에서 더 빠르게 최소값에 도달할수있다.  
#### Subtract mean
data set 의 평균 $$\mu$$을 구하고 전체 data 에 $$-\mu$$ 를 하여 0을 평균으로 하도록 변형한다.  
$$
\mu=\frac{1}{m}\sum_{i=1}^{m}x^{i}\\
x:=x-\mu\\
$$  
  
#### Normalize variance
밀집된 data 를 분산시킨다.  
$$
\sigma^{2}=\frac{1}{m}\sum_{i=1}^{m}(x^{i})^2\\
x:=\frac{x}{\sigma^{2}}\\
$$  

### Vanishing / Exploding gradients
derivatives, 기울기가 사라지거나 폭발적으로 증가하는 경우  
다음같은 방식이기 때문에 layer 가 깊어질수록 발생하기 쉽다.  
$$W^{[L]}(W^{[L-1]}(W^{[L-2]}\cdots(W^{[2]}(W^{[1]}X)))))\\$$


#### Random initialization
$$n$$ : 뉴런으로 들어가는 입력특성의 개수  
$$n\uparrow$$ 일수록 $$w_{[i]}\downarrow$$ 이어야 vanishing, exploding 방지에 효과적이다.  

$$variance(w_{i})=\frac{1}{n}$$   

$$
\text{Relu}\ (initialzation)\\
w^{[i]}=np.random.randn(shape)*np.sqrt(\frac{2}{n^{[l-1]}})\\
$$  
$$
\tanh\ (Xavier\ initialzation)\\
w^{[i]}=np.random.randn(shape)*np.sqrt(\frac{1}{n^{[l-1]}})\\
$$  

### Gradient checking
back propagation 과정에서 산출한 $${\partial \theta}^{[i]}$$와 $$f^{\prime}(\theta)$$ 를 비교하여 오류없이 진행되고 있는지 체크한다.  
$$
{\partial \theta}^{[i]}\approx{\partial \theta_approx}^{[i]}\\
$$  

$${\partial \theta}^{[i]}=\frac{\partial J}{\partial \theta_{i}}\\$$  
$$
{\partial\theta_approx}^{[i]}=\frac{J(\theta_{1},\theta_{2},\cdots,\theta_{i}+\epsilon)-{J(\theta_{1},\theta_{2},\cdots,\theta_{i}-\epsilon)}}{2\epsilon}\\
$$  
  
$$
\text{Gradient cheking } (\text{if } \epsilon=10^{-7})\\
\frac{\left\lVert \partial \theta_approx-\partial \theta \right\rVert_2}{\left\lVert \partial \theta_approx \right\rVert_2+\left\lVert \partial \theta \right\rVert}_2
\approx 
\begin{cases}
great & 10^{-7}\\ & 10^{-5}\\
bad & 10^{-3}\\
\end{cases}\\
$$  

## Optimization
### Mini batch
batch 를 여러개로 나누어 학습하는 방식  
1<= mini batch size <= batch size (사실, batch 도 mini batch 에 포함)  
1 epoch (training set 전체를 한번 학습)  

mini batch size 가 작으면 vectorization 효과가 없어서 속도가 느려지고, 크면 iteration 마다 긴 시간이 소요된다.  

training set 이 작을 경우(<=2000) batch 를 사용한다.  
CPU/GPU memory 를 초과하지 않는 mini-batch 사용한다.  
mini-batch 의 크기는 $$2^{n}$$ 으로 사용한다. (컴퓨터 처리방식이 속도에 영향)  

mini batch gradient descent 중에는 mini batch 간의 차이로 cost 에 noise 가 발생하지만, batch gradient descent 의 iteration 은 학습중 한번이라도 cost 가 상승하는 일이 있으면 이상이 있다.  

### Exponentially weighted averages
최근 데이터에 가중치를 두어 현재 데이터에 반영하여 보정하는 방법  
momentum, RMSprop, Adam 에 사용한다.  
$$V_{t}=\beta V_{t-1}+(1-\beta)\theta_{t}\\$$  

과거의 데이터는 영향이 작아지도록 설계되어 있다.  

$$
V_{99}=0.9V_{98}+0.1\theta_{99}\\
V_{98}=0.9V_{97}+0.1\theta_{98}\\
V_{97}=0.9V_{96}+0.1\theta_{97}\\
\vdots\\
$$  

$$
V_{99}=0.1\theta_{99}+0.9(0.1\theta_{98}+0.9(0.1\theta_{97}+0.9(\cdots)))\\
= (0.1\cdot\theta_{99})+(0.1\cdot(0.9)^{1}\cdot\theta_{98})+(0.1\cdot(0.9)^{2}\cdot\theta_{97})\cdots
$$  

$$\frac{1}{1-\beta}$$ 번 이전의 데이터는 영향이 없어진다. (급격히 작아진다.)  
$$
(1-\epsilon)^{\frac{1}{\epsilon}}=\frac{1}{e}\\
\beta^{\frac{1}{1-\beta}}\approx0.9^{10}\approx0.98^{50}\approx\frac{1}{e}\\
$$  

#### Bias correction
exponentially weighted averages 를 더 정확하게 계산하는 기법  
$$V_{0}=0$$ 로 설정하기 때문에 발생하는 초기 bias 를 제거해준다. (후기로 갈수록 $$\beta^{t}$$가 작아져서 원래 데이터와 일치하게 된다.)  
초기 한정 문제이기 때문에 보통 bias 보정에 신경쓰지 않는다.  
$$V_{t}=\frac{V_{t}}{1-\beta^{t}}$$  

### Momentum
gradient descent 진행 방향을 조정해준다.  
momentum 이 없는 gradient descent 보다 거의 항상 더 잘 작동한다.  
$$
v_{dW}=\beta v_{dW} + (1-\beta)dW\\
v_{db}=\beta v_{db} + (1-\beta)db\\
W:=W-\alpha v_{dW}\\
b:=b-\alpha v_{db}\\
$$  

### RMSprop (Root Mean Square Propagation)
gradient descent 진행 속도를 조정해준다.  
변동이 클수록 값이 작아진다.  
$$
s_{dW}=\beta s_{dW} + (1-\beta)dW^{2}\\
s_{db}=\beta s_{db} + (1-\beta)db^{2}\\
W:=W-\alpha \frac{dW}{\sqrt{s_{dW}}+\epsilon}\\
b:=b-\alpha \frac{dW}{\sqrt{s_{dW}}+\epsilon}\\
$$  

### Adam (Adaptive Moment Estimation)
momentum 과 RMSprop 를 합친 것 (+ bias correction)
$$
v_{dW}=\beta_{1} v_{dW} + (1-\beta_{1})dW\\
v^{corrected}_{dW}=\frac{v_{dW}}{1-(\beta_{1})^t}\\
s_{dW}=\beta_{2} s_{dW} + (1-\beta_{2})dW^{2}\\
s^{corrected}_{dW}=\frac{s_{dW}}{1-(\beta_{2})^t}\\
W:=W-\alpha \frac{v^{corrected}_{dW}}{\sqrt{s^{corrected}_{dW}}+\epsilon}\\
$$  

hyperparameter 값 (대부분 튜닝하지 않고 사용하는 값)  
$$
\beta_{1}=0.9\\
\beta_{2}=0.999\\
\epsilon=10^{-8}\\
$$  

### Learning Rate Decay
learning rate 를 조금씩 줄여가는 방법  
$$\alpha$$ 가 고정되어 있으면 최소값 근처 지점까지 도착할수는 있어도 수렴하지는 못한다.  
초기 구간에사는 빠르게, 후기 구간에서는 느리게 gradient descent 를 진행하기 위해 learning rate 를 줄여간다.  
$$\alpha=\frac{1}{1+\text{decay-rate}*\text{epoch-num}}\alpha_{0}$$  
$$
\alpha=0.95^{epoch\ num}\alpha_{0}\\
\alpha=\frac{k}{\sqrt{epoch\ num}}\alpha_{0}\\
\alpha:=\frac{\alpha}{2}\\
$$  
manual decay(수작업)  
여러가지 방식이 있다.  

### Local Optima
deep learning 초기에는 gradient descent 과정에서 local optima 에 걸리는것을 우려했다.  
하지만 실제로는 저차원 그래프와 달리 고차원의 경우 local optima 에 도달할 확률은 낮고, saddle point 에 도달할 확률이 높다고 한다.  
saddle point 의 plateaus 에서는 gradient 가 0에 가까워서 학습 속도가 느려진다. 이 상황을 벗어나는데 Adam 이 효과적이다.  

### Hyperparameter Tuning
hyperparamter 를 튜닝에서 우선순위가 있다. 사람마다 다를 수 있는데, Andrew Ng 교수는 다음과 같은 순서로 튜닝한다고 한다. (Adam 은 튜닝하지 않는다고 한다.)  

$$
\color{red} {\alpha}\\
\color{orange}{
\#hidden\ unit\\
mini\ batch\ size\\
\beta=0.9\ (momentum)\\
}
\color{purple}{
\#layers\\
learning\ rate\ decay\\
}
\beta_{1}=0.9, \beta_{2}=0.999, \epsilon=10^{-8}\ (Adam)\\
$$  

sampling 할때 어떤 hyperparameter 의 영향이 클지 알수없기 때문에 random 하게 sampling 하는게 좋다.  
그리고 random 하게 sampling 한 결과에서 성능이 높게 나온 영역을 다시 집중적으로 sampling 하여 최적의 hyperparameter 를 찾는다.  

\#hidden unit, \#layer 같은 경우 범위 내에서 random 하게 sampling  
learning rate 같이 0.0001~1 처럼 분포하는 경우, random 하게 하면 0.1~1 사이의 데이터가 대부분이기 때문에 $$\log$$ 등의 방식으로 해결한다.  
$$\beta$$ 의 경우는 $$1-\beta$$ 방식으로 sampling 하여 효율적이게 한다.  

Pandas vs. Caviar  
- pandas (babysitting one model)
  - 데이터가 크고 컴퓨팅 자원은 적을때 쓰는 방식
  - 학습을 진행하며 hyperparameter 를 변경해가며 최적화
- caviar (training many models in parallel)
  - 컴퓨팅 자원이 많을때 쓰는 방식
  - 동시에 여러모델을 실험하고 가장 잘 작동하는 모델을 선택

### Batch Normalization
hidden layer 에서 normalization 을 하는 것 (주로 Z)  
기존의 normalization 평균이 0, ±1 영역에 한정되기 때문에 $$\gamma, \beta$$ 를 사용한다. (weights 처럼 학습하는 parameter 이다.)  
batch norm 사용시 평균을 전체에서 빼기 때문에 b 는 무의미해진다.  

$$
\mu=\frac{1}{m}\sum_i^{}z^{(i)}\\
\sigma^{2}=\frac{1}{m}\sum_i^{}(z^{(i)}-\mu)^{2}\\
z^{(i)}_{norm}=\frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}\\
\tilde{z}^{(i)}=\gamma z^{(i)}_{norm}+\beta\\
$$  

covariate shift(데이터 분포가 변하는 경우) 입력 X의 분포가 변하는 경우 다시 학습시켜야 할 수 있다.  
batch norm 은 hidden unit 의 분포가 변하여 다음층에 영향을 주는것을 제한한다.  
각 layer 의 coupling 을 약화시켜 각 layer 를 독립적 학습하게 해준다.  
  
mini-batch 에서 batch norm 사용시 약간의 regularization 효과도 발생한다.(noise 감소)  
의도치 않은 효과이기 때문에 regularization 목적으로는 사용하지 말것.  
  
batch norm 에는 $$\mu, \sigma^{2}$$ 가 필요하지만, (test, 실제 사용시)1개의 데이터로는 구할수 없다.  
주로 exponentially weighted averages 를 사용  
최종 network 로 전체 training set 에서 얻은 값 사용도 가능  
z의 평균과 편차를 구하는 방법이면 왠만하면 잘 동작할거이라고 한다.  

## Deep Learning Frameworks
framework 선택 기준  
- 프로그래밍 용이성
- 속도
- 오픈소스
