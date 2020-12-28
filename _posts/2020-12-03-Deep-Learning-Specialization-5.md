---
title: "Deep Learning Specialization 5"

categories:
    - Deep Learning

tags :
    - MOOC
    - Coursera
    - Deep Learning

---

## [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models)
Coursera Deep Learning Specialization 5  
[Coursera Certificate](https://coursera.org/share/b961d17da323141e4f01f34c92bc7b50)  

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'></script>

## RNN (Recurrent Neural Networks)
  
기존의 network 구조로는 입출력 길이가 달라지는 sequence 데이터에 적용할수없다. 또한 input 위치에 맞추어 학습하기 때문에 feature 공유가 불가능하다.  
  
network 의 이전 값인 $$a^{< t-1>}$$ 을 사용해 sequence data 의 특성을 활용한다.  
$$
a^{< t>}=g(W_{aa}a^{< t-1>}+W_{ax}x^{< t>}+b_a)\\
\hat y^{< t>}=g(W_{ya}a^{< t>}+b_y)\\
$$  

$$[a^ {< t-1>},x^ {< t>}]$$ 는 concatenate 로 이루어진다.  
$$
a^{< t>}=g(W_{a}[a^ {< t-1>},x^ {< t>}]+b_a)\\
\hat y^{< t>}=g(W_{y}a^{< t>}+b_y)\\
$$  

### Loss
$$ L^{< t>}(\hat y^{< t>}, y^{< t>})=-y^{< t>}\log \hat y^{< t>}-(1-y^{< t>})\log(1-\hat y^{< t>})\\ $$  

### Model
sequence model 에는 x(input) 와 y(output) 에 따라 여러가지 타입이 있다.  

- many-to-many
  - input : sequence
  - output : sequence
- many-to-one
  - input : sequence
  - output : other
- one-to-many
  - input : other
  - output : sequence
- many(n)-to-many(m)
  - input : encoder(sequence)
  - output : decoder(sequence) 

### Language Model
NLP 에서 사용하는 단어들을 vocabulary 라는 목록으로 만들어 사용한다.  
vocabulary 에는 단어들, 문장의 끝을 의미하는 EOS token, unknown word 등이 있다.  
사용시 index 를 사용해 one-hot 형태로 변환한다.  
  
RNN 은 이전 단어들의 입력을 통해 다음 단어를 예측한다.  
$$p(y^{< T_x>}|y^{<1>},y^{<2>}\cdots y^{< T_x-1>})\\$$  
  
단어 선택은 softmax 로 얻은 확률 분포에 따라 선택한다.  
만약 가장 높은 확률의 단어만을 선택하면 항상 같은 문장이 되는 문제가 생긴다.  
  
train 에서는 $$x^{< t>}=y^{< t-1>}$$ 처럼 이전 ground truth 를 input 으로 사용한다.  
#### Sampling
sampling 에서는 다음 단어를 예측하면서 문장을 만든다. EOS 가 나올때까지 반복하거나, 반복횟수 제한으로 문장을 끝맺는다.  

sampling 에서는 $$x^{< t>}=\hat y^{< t-1>}$$ 처럼 이전 output(예측 단어)을 input 으로 사용한다.  
#### Character-Level Language Model
문자 단위의 character-level language model 도 가능하다.  
문자 단위로 이루어지기 때문에 unknown token 문제가 없지만, 훈련 비용이 크고 단어에 비해 종속성이 낮다.  

### Gradients
#### Exploding Gradients
RNN 에서의 exploding gradients 는 gradient clipping (최대값 제한) 으로 간단하게 해결 가능하다.  
#### Vanishing Gradients
RNN 은 long-range dependencies 에 취약하여 이전 정보를 잘 유지하지 못한다.  

### GRU (Gated Recurrent Unit)
> On the properties of neural machine translation: Encoder-decoder approaches

> Empirical evaluation of gated recurrent neural networks on sequence modeling

이전 정보를 잘 유지하지 못하는 점을 해결하기 위해 memory cell 을 사용한다.  
memory cell 은 이전 정보를 기억하는 작은 memory 이다.  
  
gate $$\Gamma_u$$ 를 사용하여 memory cell 을 언제 update 할지 결정한다.  
$$\Gamma_u$$(update) 는 sigmoid 로 0~1 의 값을 가진다.  
$$\Gamma_u\approx 0$$ 이거나 $$\Gamma_u\approx 1$$ 이 되어 vanishing gradients 가 발생하지 않고 정보가 유지된다. (update-1, else-0)  
  
gate $$\Gamma_r$$(relevance) 은 $$c^{< t-1>}$$이 다음 후보 $$\tilde c^{< t-1>}$$ 에 얼마나 관련이 있는지 나타낸다.  
$$
\tilde c^{< t>}=\tanh(W_c[c^ {< t-1>},x^{< t>}]+b_c)\\
\Gamma_u=\sigma(W_u[c^ {< t-1>},x^{< t>}]+b_u)\\
\Gamma_r=\sigma(W_r[c^ {< t-1>},x^{< t>}]+b_r)\\
c^{< t>}=\Gamma_u * \tilde c^{< t>}+(1-\Gamma_u)*c^{< t-1>}\\
$$  

### LSTM (Long Short Term Memory)
> Long short-term memory

GRU 보다 많은 gate 를 사용한다.  
gate - $$\Gamma_u$$(update), $$\Gamma_f$$(forget), $$\Gamma_o$$(output)  
$$
\tilde c^{< t>}=\tanh(W_c[a^ {< t-1>},x^{< t>}]+b_c)\\
\Gamma_u=\sigma(W_u[a^ {< t-1>},x^{< t>}]+b_u)\\
\Gamma_f=\sigma(W_f[a^ {< t-1>},x^{< t>}]+b_f)\\
\Gamma_o=\sigma(W_o[a^ {< t-1>},x^{< t>}]+b_o)\\
c^{< t>}=\Gamma_u * \tilde c^{< t>}+(1-\Gamma_f)*c^{< t-1>}\\
a^{< t>}=\Gamma_o * \tanh(c^{< t>})\\
$$  

### BRNN (Bidirectional RNN)
양방향으로 RNN 으로 이전, 이후 데이터를 모두 고려한 결과를 얻을수있다.  
$$\hat y^{< t>}=g(Wy[ \overrightarrow{a^{< t>}},\overleftarrow{a^{< t>}}]+b_y)\\$$  

### Deep RNN
layer 를 깊게 한 RNN, GRU, LSTM, BRNN 등을 만들수있다.  
학습에 많은 시간이 필요하다.  



## Word Embeddings

### Word Representation
> Linguistic regularities in continuous space word representations

one-hot 표현으로는 단어 사이의 관계를 일반화 할수없는 문제점이있다.  
embedding 은 단어들의 특징을 학습하여 문제를 해결한다. (face encoding 과 유사한 아이디어)  
#### Analogies
word embedding 은 다음과 같은 특성이있다.  
$$e_{man}-e_{woman} \approx e_{king}-e_{queen}\\$$  
#### t-SNE
> Visualizing Data using t-SNE

고차원 데이터를 2D 로 mapping 하는 알고리즘  
비선형적이기 때문에 t-SNE 시각화로 단어간의 추론해서는 안된다.  
#### Cosine Similarity
유사도 $$sim(u,v)$$는 vector $$u$$와 $$v$$ 사이의 각도의 cosine 값이다.  
$$
sim(e_{w}, e_{king}-e_{man}+e_{woman})\\
sim(u,v)=\frac{u^Tv}{\left\lVert u \right\rVert_2\left\lVert v \right\rVert_2}\\
$$  
#### Embedding matrix
embedding matrix $$E$$ 와 one-hot vetor(word) $$O$$ 로 word $$j$$의 embedding 을 얻는다.  
$$E\cdot O_j = e_j\\$$  
one-hot vector 의 대부분이 0 이므로 낭비가 많다. 따라서 실제로 전체 계산을 하지는 않는다.  

### Word2Vec
#### Skip-gram
> Efficient estimation of word representations in vector space

input(context) 으로 output(target) 을 예측하는 방법  
$$
y=target\\
\hat y=softmax(e_c)\\
L(\hat y, y)=-\sum_i y_i \log \hat y_i\\
$$  

#### Hierarchical Softmax
vocab 가 커질수록 softmax 의 연산이 많아지는 문제가 있다.  
$$p(t|c)=\frac{e^{\theta_t^Te_c} }{\sum_j e^{\theta_j^Te_c} }\\$$  
  
hierarchical softmax 는 전체 vocab 의 softmax 를 합산하는 연산을 하지않아서 효율적이다.  
#### Negative Sampling
> Distributed representations of words and phrases and their compositionality

binary classification 를 사용하여 softmax 보다 효율적이게 한다.  
x(context, word), y(target) 구성  

|x(context - word)|y(target)|
|------|---|
|orange - juice|1|
|orange - king|0|
|orange - book|0|
|orange - ......|0|  

1. 문장에서 context, word 를 선택하고 y 를 1로 설정  
1. vocab 에서 k(large dataset:2~5, small dataset:5~20)개의 word 를 선택, y를 0으로 설정  
1. 각각의 x,y 에 대해 k+1 번의 binary classification

random 하게 word 선택시 영어 단어의 분포를 반영하지 못하고, 사용빈도에 따라 선택하면 the, of, a, and 등의 특정 단어에 집중된다.  
따라서 다른 heuristics 을 사용한다.  
$$p(w_i)=\frac{f(w_i)^\frac{3}{4} }{\sum_j f(w_j)^\frac{3}{4} }\\$$  

### GloVe
> Glove: Global vectors for word representation

i(context) 에 j(target) 가 나타나는 횟수 $$X_{ij}$$ 를 사용한다.  
(i, j 의 정의에 따라 $$X_{ij}=X_{ji}$$ 일수도 있다.)  
  
$$\text{minimize} \sum_i \sum_j f(X_{ij})(\theta^T_ie_j+b_i+b^\prime_j-\log X_{ij})^2\\$$  
  
$$f(X_{ij})$$는 $$X_{ij}$$가 0일 경우 0으로 만든다. 또한 word 의 빈도에 따라 비중을 조정한다.  

### Debiasing
> Man is to computer programmer as woman is to homemaker? debiasing word embeddings

word embeddings 은 학습 data 에 영향을 받아서 bias(성별, 인종, 나이 등)가 생기는 문제가 있다.  
이러한점을 해결해야한다.  



## Sequence models & Attention mechanism

### Sequence to Sequence
> Sequence to sequence learning with neural networks

> Learning phrase representations using RNN encoder-decoder for statistical machine translation

### Image Captioning
CNN 의 마지막 FC layer 를 softmax layer 대신, RNN 의 input 으로 사용하여 image captioning 을 만들수있다.  

> Deep captioning with multimodal recurrent neural networks

> Show and tell: A neural image caption generator

> Deep visual-semantic alignments for generating image descriptions

### Machine Translation
machine translation 에서는 encoding, decoding 구조의 RNN 을 사용한다.  
encoding 은 입력을, decoding 에서는 출력을 처리한다.  
  
번역의 정확도를 높이기 위해서는 $$P(y^{<1>},...,y^{< T_y>}\mid x)\\$$ 을 극대화 해야한다.  

### Beam Search
greedy algorithm 으로 예측하면 전체 문장의 정확도를 고려하지 못한다.  
이를 해결하기위해 beam search 를 사용한다.  
beam search 는 parameter B(beam width) 를 사용한다.  
(B↑ 정확도↑,속도↓), (B↓ 정확도↓,속도↑)  

1. 입력 단어(문장) 다음 단어 예측(softmax)
1. 전체 확률 (B*vocab 개) 중 가장 높은 B개를 입력 단어(문장)로 선택
1. EOS 가 나올때까지 반복

$$
P(y^{<1>},\cdots,y^{< T_y>}\mid x)=
P(y^{<1>}\mid x)\cdot 
P(y^{<2>}\mid x,y^{<1>})\cdots
P(y^{< T_y>}\mid x, y^{<1>},..., y^{< T_y-1>})
$$  

확률(0~1)을 반복적으로 곱하면 값이 매우 작아지기 때문에 $$\log$$를 사용한다.  
$$
\text{arg max y} \prod_{t=1}^{T_y}P(y^{< t>}\mid x,y^{<1>},...,y^{< t-1>})\\
\text{arg max y} \sum_{t=1}^{T_y}\log P(y^{< t>}\mid x,y^{<1>},...,y^{< t-1>})\\
$$  

마찬가지로 normalization 을 사용하면 긴 문장에 효과적이다.  
$$
\frac{1}{T_y^\alpha}\sum_{t=1}^{T_y}\log P(y^{< t>}\mid x,y^{<1>},...,y^{< t-1>})\\
$$  

사람이 작성한 문장에 대한 $$P(y^* \mid x)$$ 와 machine translation 에 대한 $$P(\hat y \mid x)$$ 을 비교하여 error analysis 가 가능하다.  
$$P(y^* \mid x) > P(\hat y \mid x)$$ 이면 beam search 실패,  
$$P(y^* \mid x) < P(\hat y \mid x)$$ 이면 RNN model 문제  

#### Bleu Score
> BLEU: a method for automatic evaluation of machine translation

machine translation 의 평가 점수를 자동을 계산하는 방법  
사람이 작성한 reference 들과 machine translation 에서 일치하는, word, bigram 의 수를 평가하는 방식이다.  

### Attention Model
> Neural machine translation by jointly learning to align and translate

> Show, attend and tell: Neural image caption generation with visual attention

encoder->decoder 구조는 많은 정보(긴 문장)을 처리하기 힘들다.  
별도의 BRNN 을 만들고 output 을 decoder 의 input 으로 추가한다.  

$$
a^{< t^\prime>}=(\overrightarrow{a^{< t^\prime>} }, \overleftarrow{a^{< t^\prime>} })\\
e^{< t,t^\prime>}=NN(s^{< t-1>}, a^{< t^\prime>})\\
\alpha^{< t, t^\prime>}=\frac{exp(e^{< t,t^\prime>})}{\sum^{T_x}_{t^\prime=1}exp(e^{< t,t^\prime>})}\\
c^{< t>}=\sum_{t^\prime}\alpha^{< t,t^\prime>}a^{< t^\prime>}\\
$$


### Speech Recognition
> Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks

input(audio) 와 1:1 로 output(text)  
ex) (the quick ...) (t t t t _ _ _ h _ e e e _ _ - _ _ _ q _ _)  

#### Trigger Word Detection
label 을 text 가 아닌 0(other), 1(trigger) 로 지정한다.  

