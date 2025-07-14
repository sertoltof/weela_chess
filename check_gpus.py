import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Num GPUs Available: ", len(gpus))
    for gpu in gpus:
        print(gpu)
else:
    print("No GPUs available.")