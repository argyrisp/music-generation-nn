import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import mido
import cv2
import os

dataset_selection = "classic"


################################################################################################
if not os.path.exists("evaluation_plots"):
    os.mkdir("evaluation_plots")

if not os.path.exists("evaluation_plots\\"+dataset_selection):
    os.mkdir("evaluation_plots\\"+dataset_selection)


def load_images(path):
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
    # images = np.expand_dims(images, axis=-1)
    # images = images.astype('float32')
    # images = images / 127.5 - 1
    return images


def analysis(images):
    note_frequency = np.zeros(96)

    # xh = images[0]
    # print(len(xh))
    # print(xh.shape)
    # print(len(xh[0]))
    for i in images:
        for x in range(len(i)):
            for y in range(len(i[0])):
                if i[x][y] != 0:
                    note_frequency[95 - x] = note_frequency[95 - x] + 1

    '''for x in range(len(xh)):
        for y in range(len(xh[0])):'''

    # for i in range(len(images)):
    return note_frequency


def save_bar_graphs():
    dataset_images = load_images(path="datasets\\" + dataset_selection)
    dataset_frequency = analysis(dataset_images)
    dataset_frequency = dataset_frequency*100 / (len(dataset_images)*96)

    generated_images = load_images(path="results\\images\\"+dataset_selection)
    gen_frequency = analysis(generated_images)
    gen_frequency = gen_frequency*100 / (len(generated_images)*96)

    spacing_bar = np.zeros(96)

    fig = plt.subplots(figsize=(18, 10))
    # sizz = 14
    '''plt.rc('font', size=sizz)  # controls default text sizes
    plt.rc('axes', titlesize=sizz)  # fontsize of the axes title
    plt.rc('axes', labelsize=sizz)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sizz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=sizz)  # fontsize of the tick labels
    plt.rc('legend', fontsize=sizz)  # legend fontsize
    plt.rc('figure', titlesize=sizz)  # fontsize of the figure title'''
    barWidth = 0.25
    br1 = np.arange(len(dataset_frequency))
    br2 = [x + barWidth for x in br1]
    # br3 = [x + barWidth for x in br2]

    plt.bar(br1, dataset_frequency, color='r', width=barWidth,
            edgecolor='grey', label='Dataset Note Frequency')
    plt.bar(br2, gen_frequency, color='b', width=barWidth,
            edgecolor='grey', label='Generated Music Note Frequency')
    # plt.bar(br3, spacing_bar, color='g', width=barWidth, edgecolor='grey')
    plt.xlabel('Note', fontweight='bold', fontsize=20)
    plt.ylabel('Note Frequency %', fontweight='bold', fontsize=20)
    plt.ylim(0, 18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("evaluation_plots\\"+dataset_selection+"\\Bar_graph_GAN_"+dataset_selection+".png")

    plt.close()

    fig = plt.figure(figsize=(18, 10))
    # sizz = 14
    '''plt.rc('font', size=sizz)  # controls default text sizes
    plt.rc('axes', titlesize=sizz)  # fontsize of the axes title
    plt.rc('axes', labelsize=sizz)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sizz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=sizz)  # fontsize of the tick labels
    plt.rc('legend', fontsize=sizz)  # legend fontsize
    plt.rc('figure', titlesize=sizz)  # fontsize of the figure title'''
    plt.bar(range(96), abs(dataset_frequency-gen_frequency), color='maroon', width=0.4)
    plt.xlabel("Note", fontweight='bold', fontsize=20)
    plt.ylabel("Frequency %", fontweight='bold', fontsize=20)
    plt.ylim(0, 18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Frequency difference", fontsize=20)
    plt.savefig("evaluation_plots\\"+dataset_selection+"\\Difference_Bar_graph_GAN_"+dataset_selection+".png")
    # plt.show()
    plt.close()


save_bar_graphs()


'''# creating the bar plot
plt.bar(range(96), note_frequency, color='maroon', width=0.4)
plt.xlabel("Note")
plt.ylabel("Frequency %")
plt.title("Dataset Note Frequency")
plt.show()
plt.close(fig)





# creating the bar plot
plt.bar(range(96), note_frequency, color='maroon', width=0.4)
plt.xlabel("Note")
plt.ylabel("Frequency %")
plt.title("Generated Music Note Frequency")
plt.show()'''


def plot_losses():
    # dataset_selection = "pmd"
    path = "loss_logs\\"+dataset_selection+"\\loss_logs.txt"
    epoch = 0
    loss_f = open(path, "r")
    flag = True
    avg_loss_y = []
    val_loss_y = []
    x = []
    d_sum = []
    g_sum = []
    d_loss = []
    g_loss = []
    for line in loss_f:
        if flag:
            flag = False
            continue
        # print(line)
        d, g = line.split(",")
        if d == "e":
            epoch = g
            if len(d_sum) != 0:
                d_loss.append(sum(d_sum) / len(d_sum))
                g_loss.append(sum(g_sum) / len(g_sum))
                x.append(int(epoch) + 1)
            d_sum = []
            g_sum = []
        else:
            d_sum.append(float(d))
            g_sum.append(float(g))

        # avg_loss_y.append(float(avg_loss))
        # val_loss_y.append(float(val_loss))
        # epochs += 1

        # x.append(epochs)

    fig = plt.figure()
    sizz = 14
    plt.rc('font', size=sizz)  # controls default text sizes
    plt.rc('axes', titlesize=sizz)  # fontsize of the axes title
    plt.rc('axes', labelsize=sizz)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sizz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=sizz)  # fontsize of the tick labels
    plt.rc('legend', fontsize=sizz)  # legend fontsize
    plt.rc('figure', titlesize=sizz)  # fontsize of the figure title
    plt.plot(x, d_loss, "r", label='Discriminator Loss')
    plt.plot(x, g_loss, "b", label='Generator Loss')
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    # plt.ylim(0, 0.7)
    plt.title("Loss history")
    plt.legend()
    # plt.show()
    plt.savefig("loss_logs\\"+dataset_selection+'_losses')

    path = "loss_logs\\" + dataset_selection + "\\disc_acc.txt"
    # epoch = 0
    loss_f = open(path, "r")
    flag = True
    r_acc = []
    f_acc = []
    x = []

    for e, line in enumerate(loss_f):
        if flag:
            flag = False
            continue
        real, fake = line.split(",")
        r_acc.append(float(real))
        f_acc.append(float(fake))
        x.append(e)

    fig = plt.figure()
    sizz = 14
    plt.rc('font', size=sizz)  # controls default text sizes
    plt.rc('axes', titlesize=sizz)  # fontsize of the axes title
    plt.rc('axes', labelsize=sizz)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sizz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=sizz)  # fontsize of the tick labels
    plt.rc('legend', fontsize=sizz)  # legend fontsize
    plt.rc('figure', titlesize=sizz)  # fontsize of the figure title
    plt.plot(x, r_acc, "r", label='Discriminator Accuracy (Real)')
    plt.plot(x, f_acc, "b", label='Discriminator Accuracy (Fake)')
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy %")
    # plt.ylim(0, 0.7)
    plt.title("Discriminator Accuracy on real/fake samples")
    plt.legend()
    # plt.show()
    plt.savefig("loss_logs\\" + dataset_selection + '_acc')




plot_losses()

