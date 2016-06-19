
the follow models are using feature cube(7*7*1024) extracted from cnn (ie. GoogleNet) as an input. These model files have at least two funtions:

    init_params: init model parameters
    build_model: build the model graph,and define cost function

 
**base line model**:
 - avg_fnn: fnn means full connected layer
 - avg_lstm
 - avg_multiChannel: multiChannel means scene feature and motion featue
 
**model using focus/saliency/attention**:
 - saliency_lstm
 - saliencyLSTM_multiChannel: using LSTM in focus process
 - saliencyFgbg_multiChannel: using Fgbg(inspired from foreground background segmentation) in focus process
 
 ----------
 cnn_fnn directly uses video frames as an input, through convolutional network, full connected layer, and softmax layer.