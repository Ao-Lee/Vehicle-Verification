from keras.optimizers import Adam
from data import CarvanaReader, TripletGenerator
from mymodel import GetModel
import cfg

if __name__=='__main__':
    reader_tr = CarvanaReader(dir_images='E:\\DM\\Udacity\\Carvana\\Data\\aligned\\train')
    reader_te = CarvanaReader(dir_images='E:\\DM\\Udacity\\Carvana\\Data\\aligned\\test')
    
    gen_tr = TripletGenerator(reader_tr)
    gen_te = TripletGenerator(reader_te)
    embedding_model, triplet_model = GetModel()
    # embedding_model, triplet_model = GetModel(path=path_model)
    
    lr = 0.1
    triplet_model.compile(loss=None, optimizer=Adam(lr))

    history = triplet_model.fit_generator(gen_tr, 
                              validation_data=gen_te,  
                              epochs=20, 
                              verbose=1, 
                              workers=4,
                              steps_per_epoch=50, 
                              validation_steps=10)
    

    
    
    embedding_model.save_weights(cfg.path_model)
    for layer in embedding_model.layers:
        layer.trainable = True
        
    lr = 0.00003
    triplet_model.compile(loss=None, optimizer=Adam(lr))
    
    history = triplet_model.fit_generator(gen_tr, 
                              validation_data=gen_te,  
                              epochs=20, 
                              verbose=1, 
                              workers=4,
                              steps_per_epoch=50, 
                              validation_steps=10)
    
    embedding_model.save_weights(cfg.path_model)
    

    