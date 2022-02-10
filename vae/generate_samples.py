import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import mido
import os


dataset_selection = 'classic'


if dataset_selection == 'pmd':
    l_range = -4
    u_range = 4
    model = 'decoder_100.h5'
    print("Generating samples for "+dataset_selection+".")
elif dataset_selection == 'undertale':
    l_range = -4
    u_range = 4
    model = 'decoder_100.h5'
    print("Generating samples for "+dataset_selection+".")
elif dataset_selection == 'classic':
    l_range = -4
    u_range = 4
    model = 'decoder_100.h5'
    print("Generating samples for " + dataset_selection + ".")
else:
    l_range = 0
    u_range = 0
    model = ''
    print("Generating samples for NONE.")


def clean_image(image):
    threshold = .55
    xmax, xmin = image.max(), image.min()
    image = (image - xmin) / (xmax - xmin)
    max = image.max()
    # print(max)
    for x in range(len(image)):
        for y in range(len(image[0])):
            if image[x][y] < threshold:
                image[x][y] = 0.0
            else:
                image[x][y] = 255.0

    for x in range(len(image)):
        for y in range(len(image[0])-1):
            if y != 0 and y != len(image[0]):
                if image[x][y] == 255.0 and (image[x][y-1] == 0.0 and image[x][y+1] == 0):
                    image[x][y] = 0.0
    return image


def image2midi(image, fname):
    mid = mido.MidiFile()
    track_list = mido.MidiTrack()
    track_list.append(mido.MetaMessage(type='set_tempo', tempo=600000, time=0))
    time = 0
    for y in range(len(image[0])):
        for x in range(len(image)):
            if y != 0:
                if image[x][y] == 255 and image[x][y - 1] == 0:   # indicator that a new note is played
                    track_list.append(mido.Message(type='note_on', note=96 - x, velocity=100,
                                                   time=time))
                    time = 0
                if image[x][y] == 0 and image[x][y - 1] == 255:   # indicator a note is set off
                    track_list.append(mido.Message(type='note_off', note=96 - x, velocity=100,
                                                   time=time))
                    time = 0
        time += 1
    mid.tracks.append(track_list)
    mid.ticks_per_beat = 12
    mid.save('results\\midi\\'+dataset_selection+'\\'+fname+'.mid')
    return


def generate_song():
    r1 = random.uniform(l_range, u_range)
    r2 = random.uniform(l_range, u_range)
    sample_vector = np.array([[r1, -r2]])
    decoded_example = decoder.predict(sample_vector)


def generate_dirs():
    if not os.path.exists("results"):
        os.mkdir("results")

    if not os.path.exists("results\\images\\"):
        os.mkdir("results\\images\\")

    if not os.path.exists("results\\midi\\"):
        os.mkdir("results\\midi\\")

    if not os.path.exists("results\\images\\"+dataset_selection+"\\"):
        os.mkdir("results\\images\\"+dataset_selection+"\\")

    if not os.path.exists("results\\midi\\"+dataset_selection+"\\"):
        os.mkdir("results\\midi\\"+dataset_selection+"\\")


generate_dirs()

decoder = load_model(str('model_dir\\'+dataset_selection+'\\' + model))
print("Model loaded successfully.")
samples = 1000
for x in range(samples):
    r1 = random.uniform(l_range, u_range)
    r2 = random.uniform(l_range, u_range)
    sample_vector = np.array([[r1, -r2]])
    # print('Sample vector: ', sample_vector)
    decoded_example = decoder.predict(sample_vector)

    decoded_example_reshaped = decoded_example.reshape(96, 96)
    ''' plt.figure()
    plt.imshow(decoded_example_reshaped)
    plt.show()'''
    image = clean_image(decoded_example_reshaped)

    '''    plt.figure()
    plt.imshow(image)
    plt.show()'''
    cv2.imwrite('results\\images\\'+dataset_selection+'\\'+str(x)+"_"+str(r1)+"_"+str(-r2)+'.png', image)
    image2midi(image, str(x))
    if (x+1) % (samples/20) == 0:
        print("PROGRESS: ", x+1, "/", samples, ", [", ((x+1)/samples)*100.0, "% ]")


