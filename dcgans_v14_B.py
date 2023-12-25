# Tuning GANs network 
# save output models every 10000 epochs
# Version 13 - Compress dataset images (ok_images) using autoencoder and use as z (latent_space) for GAN-input
# Version 14 - Use Unnormalized latent_space
# Version 14A - Use latent space (from encoder) to generate the sample images
# Version 14B - Tune hyperparameters (learning_rate and beta_1 of optimizer Adam)
#--------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from tensorflow.keras import initializers
from tensorflow.keras.utils import array_to_img 
from tensorflow.keras.preprocessing import image

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

tf.config.run_functions_eagerly(True)



# Collect start time
start_prog = datetime.now()

date_time = start_prog.strftime("%d-%m-%Y %H:%M:%S")
print("Start Program =", date_time)


# Encode the original images to latent (z) size = 128
import keras
from keras import layers


encoding_dim = 32
#input_img = keras.Input(shape=(28,28,1))
input_img = keras.Input(shape=(128,128,3))


x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
opt = keras.optimizers.Adam(learning_rate=0.00004, beta_1=0.5)
autoencoder.compile(optimizer=opt, loss='binary_crossentropy')
#autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# Read input images
X_train = []
image_path = "/home/somrawee_a/project-Au/datasets/CastingProduct/casting_512x512/casting_512x512/ok_front"
#print(image_path)
image_names = os.listdir(image_path)
for i in image_names:
    #print(i)
    img = Image.open(image_path + '/' + i)
    img = img.resize((128, 128))
    img = np.asarray(img)
    X_train.append(img)

X_train = np.asarray(X_train)
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
dataset = X_train

# split data for training autoencoder

x_train, x_test = train_test_split(dataset, test_size=0.2)


autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/home/somrawee_a/project-Au/logs')])

print(x_train.shape)
print(x_test.shape)

encoder = keras.Model(input_img, encoded)
#encoded_imgs = encoder.predict(x_test)
encoded_imgs = encoder.predict(dataset)
reshaped_encoded_imgs = np.reshape(encoded_imgs, (encoded_imgs.shape[0], 128))

#init = initializers.RandomNormal(stddev=0.02)
init = initializers.TruncatedNormal(stddev=0.02, mean=0.0, seed=42)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(128, 128 ,3)),
        layers.Conv2D(32, kernel_size=3, strides=2, padding="same", kernel_initializer=init),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),

        layers.Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_initializer=init),
        layers.ZeroPadding2D(padding = ((0,1), (0,1))),
        layers.BatchNormalization(momentum=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),


        layers.Conv2D(128, kernel_size=3, strides=3, padding="same", kernel_initializer=init),
        layers.BatchNormalization(momentum=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),

        layers.Conv2D(256, kernel_size=3, strides=3, padding="same", kernel_initializer=init),
        layers.BatchNormalization(momentum=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),

        layers.Conv2D(512, kernel_size=3, strides=3, padding="same", kernel_initializer=init),
        layers.BatchNormalization(momentum=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),


        layers.Flatten(),
        layers.Dense(1, activation='sigmoid'),

    ],
    name="discriminator",
)
discriminator.summary()

latent_dim = 128
#init = initializers.RandomNormal(stddev=0.02)
init = initializers.TruncatedNormal(stddev=0.02, mean=0.0, seed=42)

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(4 * 4 * 64, kernel_initializer=init),
        layers.Reshape((4, 4, 64)),
        layers.BatchNormalization(momentum=0.8),

        layers.Conv2DTranspose(1024, kernel_size=3, strides=2, padding="same", kernel_initializer=init),
        layers.Activation("relu"),
        layers.BatchNormalization(momentum=0.8),
        #layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(512, kernel_size=3, strides=2, padding="same", kernel_initializer=init),
        layers.Activation("relu"),
        layers.BatchNormalization(momentum=0.8),
        #layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same", kernel_initializer=init),
        layers.Activation("relu"),
        layers.BatchNormalization(momentum=0.8),
        #layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", kernel_initializer=init),
        layers.Activation("relu"),
        layers.BatchNormalization(momentum=0.8),
        #layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", kernel_initializer=init),
        layers.Activation("relu"),
        layers.BatchNormalization(momentum=0.8),
        #layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(32, kernel_size=3, strides=1, padding="same", kernel_initializer=init),
        layers.Activation("relu"),
        layers.BatchNormalization(momentum=0.8),
        #layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding="same", activation="tanh", kernel_initializer=init),


    ],
    name="generator",
)
generator.summary()

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, reshaped_encoded_imgs):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.reshaped_encoded_imgs = reshaped_encoded_imgs


    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")


    @property
    def metrics(self):
        return[self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        #print('batch_size=', batch_size)
        actual_batch_size = batch_size.numpy()
        #print("Actual Batch Size:", actual_batch_size)
        random_latent_vectors0 = tf.random.normal(shape=(batch_size, self.latent_dim))
        #print('random_latent_vectos0:', random_latent_vectors0.shape)
        mean_value = np.mean(reshaped_encoded_imgs)
        std_dev = np.std(reshaped_encoded_imgs)
        ##reshaped_encoded_img = (reshaped_encoded_imgs - mean_value) / std_dev
        reshaped_encoded_img = reshaped_encoded_imgs
        #print(reshaped_encoded_img.shape)
        #print(np.min(reshaped_encoded_img))
        #print(np.max(reshaped_encoded_img))
        # Randomly select 32 values
        #random_values = np.random.choice(reshaped_encoded_img.flatten(), size=batch_size, replace=False)

        # Reshape the selected values back to the original array shape
        #random_latent_vectors = random_values.reshape((batch_size, -1))
        #random_latent_vectors = tf.convert_to_tensor(random_latent_vectors)
        selected_indices = np.random.choice(reshaped_encoded_img.shape[0], size=batch_size, replace=False)
        random_array = reshaped_encoded_img[selected_indices, :]
        random_latent_vectors = tf.convert_to_tensor(random_array)

        #print("---------------")
        #print(random_latent_vectors.shape)
        #print(random_latent_vectors[0])
        #print(tf.shape(random_latent_vectors)[0])
        #print("---------------")




        generated_images = self.generator(random_latent_vectors)

        #print('gen_images: ', generated_images.shape)
        #print('real_images: ',real_images.shape)

        combined_images = tf.concat([generated_images, real_images], axis=0)

        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        #train discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
        #    print("----prediction---")
        #    print(predictions.shape)
        #    print(predictions)
        #    print("----------------")
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        #random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        actual_batch_size = batch_size.numpy()
        #print("Actual Batch Size2", actual_batch_size)
        mean_value = np.mean(reshaped_encoded_imgs)
        std_dev = np.std(reshaped_encoded_imgs)
        ##reshaped_encoded_img = (reshaped_encoded_imgs - mean_value) / std_dev
        reshaped_encoded_img = reshaped_encoded_imgs
        #random_latent_vectors = tf.convert_to_tensor(reshaped_encoded_img)
        # Randomly select 32 values
       # random_values = np.random.choice(reshaped_encoded_img.flatten(), size=batch_size, replace=False)
        # Reshape the selected values back to the original array shape
       # random_latent_vectors = random_values.reshape((batch_size, -1))g
        selected_indices = np.random.choice(reshaped_encoded_img.shape[0], size=batch_size, replace=False)
        random_array = reshaped_encoded_img[selected_indices, :]
        random_latent_vectors = tf.convert_to_tensor(random_array)


        misleading_labels = tf.zeros((batch_size, 1))

        #train generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))



        #update loss

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


#    def save_model(self):
#        #save_path = "models"
#        save_path = "/home/somrawee_a/project-Au/results/DCGANs_v11"
#        self.discriminator.save(save_path + "/dis_dcgans_v11.h5")
#        self.generator.save(save_path + "/gen_dcgans_v11.h5")

#    def gen_images(self, image_num, latent_dim):
#        save_path = "/home/somrawee_a/project-Au/results/DCGANs_v12"
#        self.generator = keras.models.load_model(save_path + "/gen_dcgans_v12.h5", compile=False)
#        #self.generator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate = 0.0001))

#        batch_size=50
#        generated_images=[]
#        for i in range(int(image_num/batch_size)):
#            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
#            gen_images=[]

#            gen_images = self.generator(random_latent_vectors)
#            generated_images.extend(gen_images)
#            gen_images = (gen_images + 1) * 255
#            #gen_images.numpy()
#            #if i == 0 :
#            #    generated_images = gen_images
#            #    print(generated_images.shape)
#            #else:
#            #    generated_images.extend(gen_images)
#            #    print(generated_images.shape)



        #generated_images = (generated_images + 1)*255
        #generated_images.numpy()

#        for i, img in enumerate(generated_images):
#            #path = "./generated_images"
#            img = keras.preprocessing.image.array_to_img(generated_images[i])
#            img.save("/home/somrawee_a/project-Au/results/DCGANs_v12/dcgan_gen_%d.png" % (i+1))

 #   def classify_gen_images(self, images):
#        #gen_images = images
#        image_num = len(images)
#        pred_prob=[]
#        self.discriminator = keras.models.load_model("/home/somrawee_a/project-Au/results/DCGANs_v11/dis_dcgans_v12.h5", compile=False)

#        batch_size = 50
#        for i in range(int(image_num/batch_size)):
#            gen_images = images[i*batch_size:(i+1)*batch_size-1]
#            print(i*batch_size, (i+1)*batch_size-1)
#            predictions = []
#            predictions = self.discriminator(gen_images)
#            pred_prob.extend(predictions)



#        return pred_prob

#    def cal_inception_score(self, img_num, latent_dim):


#        inception_model = InceptionV3()
#        p_y_given_x = inception_model.predict_on_batch(generated_images)
#        q_y = np.mean(p_y_given_x, axis=0)
#        inception_scores = p_y_given_x * (np.log(p_y_given_x) - np.log(q_y))
#        inception_scores = np.exp(np.mean(inception_scores))
#        return inception_scores

    def plot_graph_loss(self, gen_losses, disc_losses):

        epochs = list(range(1, len(gen_losses) +1))

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, gen_losses, label='Generator Loss', marker='o')
        plt.plot(epochs, disc_losses, label='Discriminator Loss', marker='o')

        # Customize the plot
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Generator and Discriminator Loss')
        plt.legend()
        plt.grid(True)

        # Save and display the plot
        savemodelpath="/home/somrawee_a/project-Au/results/DCGANs_v14B"
        plt.savefig(savemodelpath + '/loss_plot.png')  # Save the plot as an image file
        plt.show()  # Display the plot


class GANMonitor(keras.callbacks.Callback):B
    def __init__(self, num_img=3, latent_dim=100):
        self.num_img = num_img
        self.latent_dim = latent_dim


    def on_epoch_end(self, epoch, logs=None):
        #print(list(logs))
        #print('loss %02d %.3f %.3f' %(epoch, logs['g_loss'], logs['d_loss']))
        #f.write('%02d %.3f\n' % (epoch, logs['loss']))
        list_g_loss.append(logs['g_loss'])
        list_d_loss.append(logs['d_loss'])

        if epoch % 1000 == 0:
            ##random_latent_vectors=tf.random.normal(shape=(self.num_img, self.latent_dim))
            reshaped_encoded_img = reshaped_encoded_imgs
            selected_indices = np.random.choice(reshaped_encoded_img.shape[0], size=self.num_img, replace=False)
            random_array = reshaped_encoded_img[selected_indices, :]
            random_latent_vectors = tf.convert_to_tensor(random_array)
            generated_images = self.model.generator(random_latent_vectors)
            #generated_images *= 255
            generated_images = (generated_images+1)*127.5
            generated_images.numpy()
            for i in range(self.num_img):
                #img = keras.preprocessing.image.array_to_img(generated_images[i])
                img = array_to_img(generated_images[i])
                img.save("/home/somrawee_a/project-Au/results/DCGANs_v14B/generated_img_%06d_%d.png" %(epoch, i))
            self.model.generator.save('/home/somrawee_a/project-Au/results/DCGANs_v14B/dcganv11_gen_model_%d.h5' %epoch)

        #if epoch == epochs:
            #print(self.list_d_loss)

#epochs = 100000
            
start_time = datetime.now()

date_time = start_time.strftime("%d-%m-%Y %H:%M:%S")
print("Start Time =", date_time)

epochs = 10000
list_d_loss = []
list_g_loss = []


gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim, reshaped_encoded_imgs=reshaped_encoded_imgs)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=5, latent_dim=latent_dim)]
)
savemodelpath="/home/somrawee_a/project-Au/results/DCGANs_v14B"
gan.generator.save(savemodelpath + '/final_generator_model.h5')
gan.discriminator.save(savemodelpath +'/final_discriminator_model.h5')

gan.plot_graph_loss(list_g_loss, list_d_loss)

stop_time = datetime.now()

date_time = stop_time.strftime("%d-%m-%Y, %H:%M:%S")
print("Stop Time =", date_time)


duration_time = stop_time - start_time

print(start_time)
print(stop_time)

duration_min = duration_time /timedelta(minutes=1)
print('Run Tim (Mins): ', duration_min)

duration_sec = duration_time/timedelta(seconds=1)
print(duration_sec)

duration_hour = duration_time/timedelta(hours=1)
print('Run Time (Hours): ', duration_hour)
