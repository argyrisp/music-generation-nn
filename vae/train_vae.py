import tensorflow as tf
# import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape

from tensorflow.keras.models import Model
# from keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
dataset_selection = "classic"


def load_images(path="datasets\\" + dataset_selection):
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # image.append(cv2.imread(str(path)+"\\"+filename))
            # temp_img = cv2.imread(str(path) + "\\" + filename)
            # im2show = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
            im2show = cv2.imread(str(path) + "\\" + filename, 0)
            images.append(im2show)

            # cv2.imshow("win", cv2.resize(image, (384, 384)))
            # cv2.imshow("win", im2show)
            # cv2.waitKey(0)
    images = np.expand_dims(images, axis=3)
    return images


# path = "undertale_dataset"
images = load_images()
images = images/255
x_train = images.astype('float32')
x_test = images.astype('float32')
y_test = np.zeros(len(x_test))
y_train = np.zeros(len(x_test))

img_width = x_train.shape[1]
img_height = x_train.shape[2]
num_channels = 1
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, num_channels)
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, num_channels)
input_shape = (img_height, img_width, num_channels)

'''plt.figure(1)
plt.subplot(221)
plt.imshow(x_train[32][:, :, 0], cmap='gray')

plt.subplot(222)
plt.imshow(x_train[156][:, :, 0], cmap='gray')

plt.subplot(223)
plt.imshow(x_train[8743][:, :, 0], cmap='gray')
''''''print(y_train[32])
print(y_train[156])
print(y_train[8743])''''''
plt.show()'''

# ==== Model ====
# === Encoder ===

latent_dim = 2  # number of latent dim parameters

input_img = Input(shape=input_shape, name='encoder_input')
x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)

conv_shape = K.int_shape(x)
print(conv_shape)
print(type(conv_shape))

x = Flatten()(x)
x = Dense(32, activation='relu')(x)


z_mu = Dense(latent_dim, name='latent_mu')(x)
z_sigma = Dense(latent_dim, name='latent_sigma')(x)


def sample_z(args):
    z_mu, z_sigma = args
    eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
    return z_mu + K.exp(z_sigma / 2) * eps


z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mu, z_sigma])

encoder = Model(input_img, [z_mu, z_sigma, z], name='encoder')
encoder.summary()

# === Decoder ===

decoder_input = Input(shape=(latent_dim, ), name='decoder_input')

x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2DTranspose(num_channels, 3, padding='same', activation='sigmoid', name='decoder_output')(x)

decoder = Model(decoder_input, x, name='decoder')
decoder.summary()

z_decoded = decoder(z)


class CustomLayer(tf.keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        recon_loss = tf.keras.metrics.binary_crossentropy(x, z_decoded)

        kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        return K.mean(recon_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x


y = CustomLayer()([input_img, z_decoded])

# VAE
vae = Model(input_img, y, name='vae')
vae.compile(optimizer='adam', loss=None)
vae.summary()

print("stop here")
'''logs_path = "loss_logs\\"
if not os.path.exists(logs_path):
    os.mkdir(logs_path)
ftxt = open(logs_path + "loss_logs" + ".txt", "w")
ftxt.write("losses\n")
ftxt.close()'''

if not os.path.exists("model_dir"):
    os.mkdir("model_dir")
if not os.path.exists("model_dir\\" + dataset_selection):
    os.mkdir("model_dir\\" + dataset_selection)

if not os.path.exists("results"):
    os.mkdir("results")
if not os.path.exists("results\\plots"):
    os.mkdir("results\\plots")
if not os.path.exists("results\\plots\\" + dataset_selection):
    os.mkdir("results\\plots\\" + dataset_selection)

logs_path = "results\\loss_logs\\"
if not os.path.exists(logs_path):
    os.mkdir(logs_path)
if not os.path.exists(logs_path+dataset_selection):
    os.mkdir(logs_path+dataset_selection)
ftxt = open(logs_path+dataset_selection+"\\" + "loss_logs" + ".txt", "w")
ftxt.write("losses\n")
ftxt.close()


if not os.path.exists("plots"):
    os.mkdir("plots")
if not os.path.exists("plots\\"+dataset_selection+"\\"):
    os.mkdir("plots\\"+dataset_selection+"\\")


EPOCHS = 100
for epoch in range(EPOCHS):
    print("EPOCH: "+str(epoch))
    history_callbacks = vae.fit(x_train, None, epochs=1, batch_size=32, validation_split=.2)
    loss_history = history_callbacks.history["loss"]
    val_loss_history = history_callbacks.history["val_loss"]
    print("Losses: " + str(loss_history))
    print("val loss: "+str(val_loss_history))
    # numpy_loss_history = np.array(loss_history)
    ftxt = open(logs_path+dataset_selection+"\\" + "loss_logs" + ".txt", "a")
    ftxt.write(str(loss_history[0]) + "," + str(val_loss_history[0])+"\n")
    ftxt.close()
    # np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
    if True:  # epoch % 2 == 0 or epoch == 9:
        vae.save("model_dir\\"+dataset_selection+"\\vae_%03d.h5" % (epoch + 1))
        decoder.save("model_dir\\"+dataset_selection+"\\decoder_%03d.h5" % (epoch + 1))
        plt.close()
        mu, _, _ = encoder.predict(x_test)
        # Plot dim1 and dim2 for mu
        plt.figure(figsize=(10, 10))
        plt.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='brg')
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.colorbar()
        plt.savefig('results\\plots\\'+dataset_selection+"\\"+'space_' + str(epoch + 1))

        n = 10
        grid_x = np.linspace(-2, 2, n)
        grid_y = np.linspace(-2, 2, n)[::-1]
        # examples = np.zeros((n*n, 96, 96, 1))
        examples = []
        # decoder for each square in the grid
        for i, yi in enumerate(grid_y):
            horizontal = []
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                ex = x_decoded[0].reshape(img_width, img_height)
                xmax, xmin = ex.max(), ex.min()
                ex = (ex - xmin)*255.0 / (xmax - xmin)
                horizontal.append(ex)
                # cv2.imwrite('plots\\'+dataset_selection+'\\' + str(epoch) + '_' + str(i) + '_' + str(j) + '.png', ex)
                # print(digit.shape)
                # np.append(examples, digit)
                # examples.append(digit)
                # figure[i * img_width: (i + 1) * img_width, j * img_height: (j + 1) * img_height] = digit
            im_v = cv2.hconcat(horizontal)
            examples.append(im_v)
        example_image = cv2.vconcat(examples)
        cv2.imwrite('plots\\'+dataset_selection+'\\'+'%03d.png' % (epoch + 1), example_image)
        # print(examples.shape)
        # print(examples)



# Visualize latent space
plt.close()
mu, _, _ = encoder.predict(x_test)
# Plot dim1 and dim2 for mu
plt.figure(figsize=(10, 10))
plt.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='brg')
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.colorbar()
# plt.savefig('results\\space_'+str(epoch))
plt.show()


# Visualize images
#Single decoded image with random input latent vector (of size 1x2)
#Latent space range is about -5 to 5 so pick random values within this range
#Try starting with -1, 1 and slowly go up to -1.5,1.5 and see how it morphs from
#one image to the other.
sample_vector = np.array([[1, -1]])
decoded_example = decoder.predict(sample_vector)
decoded_example_reshaped = decoded_example.reshape(img_width, img_height)
plt.imshow(decoded_example_reshaped)

#Let us automate this process by generating multiple images and plotting
#Use decoder to generate images by tweaking latent variables from the latent space
#Create a grid of defined size with zeros.
#Take sample from some defined linear space. In this example range [-4, 4]
#Feed it to the decoder and update zeros in the figure with output.


n = 20  # generate 15x15 digits
figure = np.zeros((img_width * n, img_height * n, num_channels))

#Create a Grid of latent variables, to be provided as inputs to decoder.predict
#Creating vectors within range -5 to 5 as that seems to be the range in latent space
grid_x = np.linspace(-5, 5, n)
grid_y = np.linspace(-5, 5, n)[::-1]

# decoder for each square in the grid
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(img_width, img_height, num_channels)
        figure[i * img_width: (i + 1) * img_width,
               j * img_height: (j + 1) * img_height] = digit

plt.figure(figsize=(10, 10))

# Reshape for visualization
fig_shape = np.shape(figure)
figure = figure.reshape((fig_shape[0], fig_shape[1]))

plt.imshow(figure, cmap='gnuplot2')
plt.show()

