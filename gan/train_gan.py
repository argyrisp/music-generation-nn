import numpy as np
# from tensorflow import keras
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from matplotlib import pyplot
import os
import cv2

# TODO: 128 12 12, try 32 24 24 one less upsample and then 64 24 24 or 256 12 12
dataset_selection = "classic"


def define_discriminator(in_shape=(96, 96, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 12 * 12
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((12, 12, 128)))
    # upsample to 12x12
    # model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    # model.add(LeakyReLU(alpha=0.2))
    # upsample to 24x24
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 48x48
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 96x96
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7, 7), activation='tanh', padding='same'))
    model.summary()
    return model


def define_gan(g_model, d_model):
    # freeze discriminator weights
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


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
    images = np.expand_dims(images, axis=-1)
    images = images.astype('float32')
    images = images / 127.5 - 1
    return images


def load_real_samples():
    # load mnist dataset
    (trainX, _), (_, _) = load_data()
    # expand to 3d, e.g. add channels dimension
    X = np.expand_dims(trainX, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [0,1]
    X = X / 127.5 - 1
    return X


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    # y = np.random.uniform(0.9, 1.0, size=(n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):
    # x_input = np.random.randn(latent_dim * n_samples)
    # x_input = x_input.reshape(n_samples, latent_dim)
    x_input = np.random.uniform(-1, 1, (n_samples, latent_dim))
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    # y = np.random.uniform(0.0, 0.1, size=(n_samples, 1))
    return X, y


def save_plot(examples, epoch, n=10):
    examples = 0.5 * examples + .5
    for i in range(n*n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(examples[i, :, :, 0], cmap='gray')

    if not os.path.exists("plots"):
        os.mkdir("plots")
    if not os.path.exists("plots\\"+dataset_selection+"\\"):
        os.mkdir("plots\\"+dataset_selection+"\\")

    filename = "plots\\"+dataset_selection+"\\generated_plot_e%03d.png" % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    ftxt = open(logs_path + dataset_selection + "\\disc_acc" + ".txt", "a")
    ftxt.write(str(acc_real*100)+","+str(acc_fake*100)+"\n")
    ftxt.close()
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    if not os.path.exists("model_dir"):
        os.mkdir("model_dir")
    if not os.path.exists("model_dir\\"+dataset_selection):
        os.mkdir("model_dir\\"+dataset_selection)

    filename = 'model_dir\\'+dataset_selection+'\\generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=300, n_batch=64):
    global logs_path
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        ftxt = open(logs_path + dataset_selection + "\\loss_logs" + ".txt", "a")
        ftxt.write("e," + str(i) + "\n")
        ftxt.close()
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(X, y)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss))
            ftxt = open(logs_path + dataset_selection + "\\loss_logs" + ".txt", "a")
            ftxt.write(str(d_loss)+","+str(g_loss)+"\n")
            ftxt.close()

        if (i+1) % 1 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


logs_path = "loss_logs\\"
if not os.path.exists(logs_path):
    os.mkdir(logs_path)
if not os.path.exists(logs_path+dataset_selection+"\\"):
    os.mkdir(logs_path+dataset_selection+"\\")


ftxt = open(logs_path + dataset_selection + "\\loss_logs" + ".txt", "w")
ftxt.write("d ///////////////// g\n")
ftxt.close()

ftxt = open(logs_path + dataset_selection + "\\disc_acc" + ".txt", "w")
ftxt.write("real / fake\n")
ftxt.close()


latent_dim = 100
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)

dataset = load_images()
train(g_model, d_model, gan_model, dataset, latent_dim)






