import numpy as np
import keras
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense

keras.backend.set_image_dim_ordering('th')

path_model = 'E:\\DM\\Udacity\\Models\\yolo-tiny.weights'


def _load_weights(model,yolo_weight_file):           
    tiny_data = np.fromfile(yolo_weight_file,np.float32)[4:]
    index = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights)>0:
            filter_shape, bias_shape = [w.shape for w in weights]
            if len(filter_shape)>2: #For convolutional layers
                filter_shape_i = filter_shape[::-1]
                bias_weight = tiny_data[index:index+np.prod(bias_shape)].reshape(bias_shape)
                index += np.prod(bias_shape)
                filter_weight= tiny_data[index:index+np.prod(filter_shape_i)].reshape(filter_shape_i)
                filter_weight= np.transpose(filter_weight,(2,3,1,0))
                index += np.prod(filter_shape)
                layer.set_weights([filter_weight,bias_weight])
            else: #For regular hidden layers
                bias_weight = tiny_data[index:index+np.prod(bias_shape)].reshape(bias_shape)
                index += np.prod(bias_shape)
                filter_weight= tiny_data[index:index+np.prod(filter_shape)].reshape(filter_shape)
                index += np.prod(filter_shape)
                layer.set_weights([filter_weight,bias_weight])
                
def GetModel():
    model = Sequential()
    model.add(Conv2D(16, (3, 3),input_shape=(3,448,448),padding='same',strides=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32,(3,3) ,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(64,(3,3) ,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(128,(3,3) ,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(256,(3,3) ,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(512,(3,3) ,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(1024,(3,3) ,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(1024,(3,3) ,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(1024,(3,3) ,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    _load_weights(model, path_model)
    return model