'''
hyper-params of network structure
'''
image_size = 299
batch_size = 16
embedding_dim = 512
drop_rate = 0.4 

path_model = 'E:\\DM\\Udacity\\Models\\vgg19_dim512'


'''
network: vgg19
drop_rate: 0.4 
embedding_dim: 512
auc: 0.991
accuracy: 0.974
training loss: 0.079
validation loss: 0.048
'''

'''
network: vgg16
drop_rate: 0.5 (underfitting)
embedding_dim: 256
auc: 0.995
accuracy: 0.985
training loss: 0.131
validation loss: 0.067
'''

'''
network: vgg19
drop_rate: 0.4 
embedding_dim: 256
auc: 0.989
accuracy: 0.972
training loss: 0.072
validation loss: 0.069
'''