import tensorflow as tf
import datetime
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def LoadModel(Focal = False):
    model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None)
    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    fc = tf.keras.layers.Dense(1, activation='sigmoid', name='fc')(avg_pool)
    model = tf.keras.models.Model(inputs=model.input, outputs=fc)
    optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=1e-3)
    if Focal: loss = tf.keras.losses.BinaryFocalCrossentropy()
    else: loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

if __name__=='__main__':
    epoch = 10
    batch_size = 32

    train_data = tf.keras.utils.image_dataset_from_directory("./Dataset_CvDl_Hw2_Q5/training_dataset", image_size=(224, 224), label_mode='binary', batch_size=batch_size)
    valid_data = tf.keras.utils.image_dataset_from_directory("./Dataset_CvDl_Hw2_Q5/validation_dataset", image_size=(224, 224), label_mode='binary', batch_size=batch_size)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = "logs/binary/" + current_time
    model_name = 'models/binary/binary' + current_time +'.h5'

    resnet50 = LoadModel()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    resnet50.fit(train_data, validation_data=valid_data, epochs=epoch, callbacks=[tensorboard_callback], batch_size=batch_size)
    resnet50.save(model_name)

    log_dir = "logs/focal/" + current_time
    model_name = 'models/focal/focal' + current_time +'.h5'

    resnet50 = LoadModel(Focal=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    resnet50.fit(train_data, validation_data=valid_data, epochs=epoch, callbacks=[tensorboard_callback], batch_size=batch_size)
    resnet50.save(model_name)