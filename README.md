# NLP Notes
 
### Attention

>**Additive/concat Attention**
>>**Resources**: [[Paper]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/[Attention]Neural-Machine-Translation-by-Jointly-Learning-to-Align-and-Translate.pdf),&nbsp; [[Illustrative Intro]](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3),&nbsp; [[TF2 Implementation]](https://github.com/ywu94/NLP-Notes/blob/master/Implementations/add-attn-tf2implementation.py)

>**Multiplicative Attention**
>>**Resources**: [[Paper]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/[Attention]Effective-Approaches-to-Attention-based-Neural-Machine-Translation.pdf),&nbsp; [[Illustrative Intro]](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3),&nbsp; [[TF2 Implementation]](https://github.com/ywu94/NLP-Notes/blob/master/Implementations/mul-attn-tf2implementation.py)

>**Multi-head Self Attention / Transformer**
>>**Resources**: [[Paper]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/[Attention]Attention-Is-All-You-Need.pdf),&nbsp; [[Illustrative Intro]](http://jalammar.github.io/illustrated-transformer/),&nbsp; [[TF2 Implementation]](https://github.com/ywu94/NLP-Notes/tree/master/Implementations/transformer-tf2implementation),&nbsp;
[[TF2 Implementation by Google]](https://www.tensorflow.org/tutorials/text/transformer)
 
### Industrial Application

>**Google Neural Machine Translation System**
>>**Concept applied**: `Additive/concat attention`, `Residual connection`, `Vanilla dropout` <br/>**Resources**: [[Paper]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/[Industry]Google%E2%80%99s-Neural-Machine-Translation-System.pdf),&nbsp; [[Illustrative Intro]](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3),&nbsp; [[TF2 Implementation]](https://github.com/ywu94/NLP-Notes/blob/master/Implementations/gnmt-tf2implementation.py),&nbsp;  [[Torch Implementation]](https://github.com/ywu94/NLP-Notes/blob/master/Implementations/gnmt-torchimplementation.py)

>**BERT: Bidirectional Encoder Representations from Transformers**
>>**Resources**: [[Paper]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/%5BIndustry%5DBERT-Pre-training-of-Deep%20Bidirectional-Transformers-for-Language-Understanding.pdf)

### Probabilistic Graph 

> **Conditional Random Field**
>> **Resources**: [[Introduction to CRF]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/%5BCRF%5DAn-Introduction-to-Conditional-Random-Field.pdf),&nbsp;  [[CRF vs MRF]](https://stats.stackexchange.com/questions/156697/whats-the-difference-between-a-markov-random-field-and-a-conditional-random-fie),&nbsp;  [[CRF for Multi-label Classification]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/%5BCRF%5DCollective-Multi-Label-Classification.pdf)&nbsp;  [[Tensorflow CRF]](https://www.tensorflow.org/addons/api_docs/python/tfa/text/crf);

> **Bi-LSTM CRF**
>> **Resources**: [[Paper]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/%5BNER%5DNeural-Architectures-for-Named-Entity-Recognition.pdf),&nbsp;  [[TF1.0 Implementation by Scofield]](https://github.com/scofield7419/sequence-labeling-BiLSTM-CRF)

> **Label Attention Network**
>> **Resources**: [[Paper]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/%5BNER%5DHierarchically-Refined-Label-Attention-Network-for-Sequence-Labeling.pdf),&nbsp;  [[Torch Implementation by Author]](https://github.com/Nealcly/BiLSTM-LAN)
   
### Modeling Tricks

>**Recurrent Neural Network Normalization**
>>**Resources**: [[Methodology Overview]](https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/),&nbsp; [[Layer Normalization]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/%5BRNN-Training%5DLayer-Normalization.pdf)<br/>**Experience**: use `BatchNormalization` or `LayerNormalization` after each RNN layer

>**Recurrent Neural Network Dropout**
>>**Resources**: [[Methodology Overview]](https://medium.com/@bingobee01/a-review-of-dropout-as-applied-to-rnns-72e79ecd5b7b),&nbsp; [[Vanilla Dropout]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/%5BRNN-Dropout%5DRecurrent-Neural-Network-Regularization.pdf),&nbsp; [[Variational Dropout]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/%5BRNN-Dropout%5DA-Theoretically-Grounded-Application-of-Dropout-in-Recurrent-Neural-Networks.pdf),&nbsp; [[Recurrent Dropout]](https://github.com/ywu94/NLP-Notes/blob/master/Papers/%5BRNN-Dropout%5DRecurrent-Dropout-without-Memory-Loss.pdf)<br/>**Experience**: set dropout ratio between `0.1` and `0.3`, begin with `vanilla dropout`

