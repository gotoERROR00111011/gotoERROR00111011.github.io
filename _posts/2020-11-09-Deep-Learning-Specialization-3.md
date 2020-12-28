---
title: "Deep Learning Specialization 3"

categories:
    - Deep Learning

tags :
    - MOOC
    - Coursera
    - Deep Learning

---

## [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects?specialization=deep-learning)
Coursera Deep Learning Specialization 3  
[Coursera Certificate](https://coursera.org/share/e8fc243e0172c1a64e3119cfb66ea8f7)  

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'></script>

## ML Strategy

## Orthogonalization
machine learning 시스템 구축에는 시도 할수있는 일이 많고, 바꿀수 있는 것도 많다.(hyperparameters)  


## Goal
project 의 목표를 제대로 설정하는 것이 중요하다.  

### Evaluation metric
evaluation metric 을 하나만 갖도록 하는것이 좋다. (F1 score, 평균)  
F1 Score = harmonic mean, precision(정밀도), recall(재현율)  
$$F_1=\frac{2}{recall^{-1}+precision^{-1}}\\$$  

### Satisficing and Optimizing metric
optimizing metric : 가장 좋은 결과를 원하는 1개 설정  
satisficing metric : 특정 기준을 만족시키면 pass 방식  


## Human-Level Performance
### human level performance  
사람이 직접 처리하는 수준의 성능
  
### bayesian optimal error
이론상 최대 성능  
human level performance 을 넘으면 학습속도가 더뎌진다.  
- bayesian optimal error 가 human level performance 와 가깝기 떄문
- 인간으로부터의 labeled data 제공에 한계 (인간이 잘하는 업무를 ML 이 잘하는 이유)

avoidable bias : bayes optimal error 와 training error 의 차이  


## Error Analysis
error 분석(mislabeled, blur, etc) 후 비율에 따라 필요시 수작업한다.  

### Incorrectly labeled data
training set 에서는 전체 데이터가 충분히 많다면 문제없다.  
(단, random 해야한다. 특정 label 데이터의 mislabel 이 반복된다면 문제가 있음)

### quickly, iterate
새로운 알고리즘 (논문)을 만들어 내는것이 아니라면 시스템을 빠르게 만들고 테스트를 반복하는것이 효과적이다.  


## Mismatched training and dev/test set
다양한 분포의 데이터가 있을 경우 project 의 목표에 맞추어 train, dev, test set 으로 나눈다.  
train set 과 dev/test set 의 데이터 분포가 다를 경우 error 분석을 위해 train set 과 같은 분포를 가지는 train-dev set 를 추가로 만든다.  

human level  
↕ avoidable bias  
train set error  
↕ variance  
train-dev set error  
↕ data mismatch  
dev error  
test error  

## Addressing data mismatch
artificial data synthesis 로 부족한 분포의 데이터를 보충할수있다.  

## Learning from multiple tasks
### Transfer Learning
학습된 network 를 다른 데이터셋에 사용하는 방법  
pre-training 된 network 를 새로운 data set 으로 fine tuning 하는 방식  
  
image, sound 등의 데이터가 초기 layer 에서 비슷한 특성을 가져서 가능한 방법이다.  
  
일단 출력층의 layer 의 weights 를 random 하게 초기화한다.  
새롭게 학습할 data set 이 작으면 출력 layer 만 재학습, 크면 전체 network 를 재학습 할수있다.  
output layer 에 새로운 network 를 추가로 붙이는 방법도 가능하다.  
  
data set 이 작을때, 비슷한 유형의 data set 을 활용하여 보충 가능하다.  

### Multi-task Learning
multi-task learning 은 softmax 와 다르게 하나의 이미지가 multiple labels 를 가질수있다.  
(일반적으로 각각의 label 에 대하여 logistic regression 사용)  
network 의 초기 특성을 공유하여 효과를 발휘한다.  
  
다음 조건을 충족시키면 성능 향상에 도움이 된다.  
1. 초기 특성이 유사해야한다. (image, sound)
1. label 간 비슷한 데이터 수
1. 충분히 큰 network

## End-to-End Deep Learning
data(x->y) 가 충분하다면 end-to-end 가 성능을 발휘한다.  
하지만 현재 시점에서는 개발자가 지정한 처리 pipeline 을 거치는 것이 더 효과적이다.  

기계번역 같은 경우 데이터가 매우 많기 때문에 end-to-end 방식에 적합하다.  
얼굴 인증의 경우, 얼굴위치인식 -> 얼굴인증 과정을 거쳐야 한다. (데이터가 많아지고, HW 성능이 높아진다면 end-to-end 방식이 가능해질지도 모른다.)  
