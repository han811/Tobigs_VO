# conv block1 settings
conv_block1 = dict()
conv_block1['in_channels']=3
conv_block1['out_channels']=64
conv_block1['kernel_size']=7
conv_block1['stride']=2
conv_block1['padding']=3
conv_block1['bias']=False

# maxpool settings
maxpool_block = dict()
maxpool_block['kernel_size']=3
maxpool_block['stride']=2
maxpool_block['padding']=1


# conv_block2 settings
conv_block2 = dict()
conv_block2['channel1']=64 #1x1 channel
conv_block2['channel2']=64 #3x3 channel
conv_block2['channel3']=256 #1x1 channel
conv_block2['stride']=1 #stride

# conv_block3 settings
conv_block3 = dict()
conv_block3['channel1']=128 #1x1 channel
conv_block3['channel2']=128 #3x3 channel
conv_block3['channel3']=512 #1x1 channel
conv_block3['stride']=2 #stride

# conv_block4 settings
conv_block4 = dict()
conv_block4['channel1']=256 #1x1 channel
conv_block4['channel2']=256 #3x3 channel
conv_block4['channel3']=1024 #1x1 channel
conv_block4['stride']=2 #stride

# conv_block5 settings
conv_block5 = dict()
conv_block5['channel1']=512 #1x1 channel
conv_block5['channel2']=512 #3x3 channel
conv_block5['channel3']=2048 #1x1 channel
conv_block5['stride']=2 #stride

# linear layers settings
linear_block = dict()
linear_block['layer1']=2048
linear_block['layer2']=1024

# rnn_block1 settings
rnn_block1 = dict()
# rnn_block1['input_size']=4096
# rnn_block1['input_size']=2048
rnn_block1['input_size']=2048

rnn_block1['hidden_size']=1000
rnn_block1['num_layers']=2
rnn_block1['batch_first']=True

# rnn_block2 settings
rnn_block2 = dict()
# rnn_block2['input_size']=4096
# rnn_block2['input_size']=2048
rnn_block2['input_size']=2048

rnn_block2['hidden_size']=1000
rnn_block2['num_layers']=1
rnn_block2['batch_first']=True

# fc_layer1 settings
fc1 = dict()
fc1['input']=1000
fc1['output']=1024

# fc_layer2 settings
fc2 = dict()
fc2['input']=1000
fc2['output']=1024

# fc_layer3 settings
fc3 = dict()
fc3['input']=2048
fc3['output']=1024

# fc_layer4 settings
fc4 = dict()
fc4['input']=1024
fc4['output']=3

# fc_layer5 settings
fc5 = dict()
fc5['input']=1024
fc5['output']=4

# layer_test1 settings
test1 = dict()
test1['input']=1024
test1['output']=1024

# layer_test2 settings
test2 = dict()
test2['input']=1024
test2['output']=1024

# resnet-50 parameter
resnet_blk = dict()
resnet_blk['conv1']=conv_block1
resnet_blk['maxpool']=maxpool_block
resnet_blk['conv2']=conv_block2
resnet_blk['conv3']=conv_block3
resnet_blk['conv4']=conv_block4
resnet_blk['conv5']=conv_block5
resnet_blk['linear']=linear_block
resnet_blk['rnn1']=rnn_block1
resnet_blk['rnn2']=rnn_block2
resnet_blk['fc1']=fc1
resnet_blk['fc2']=fc2
resnet_blk['fc3']=fc3
resnet_blk['fc4']=fc3
resnet_blk['fc5']=fc3
resnet_blk['test1']=test1
resnet_blk['test2']=test2