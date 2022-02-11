# Music Generation with Neural Networks
Utilizing Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) to generate musical samples from MIDI files.

This project was done as a part of my thesis research for the degree of BSc (Technical University of Crete, Greece).

In this repository you will find two approaches to this generative task, GAN and VAE. In both cases there is a dedicated script in preprocessing MIDI files converting them to images. Those images are the dataset used for training the models. Each model is trained separately, and then they are used to generate sample images that then get converted to .mid files for listening purposes. We train each model in three different piano music datasets (six total iterations), each dataset containing different musical style with varying sample quantity for comparison reasons:
- Pokemon Mystery Dungeon Red/Blue Rescue Team soundtrack (11 tracks converted to 298 images, video game ambient music with little variance between tracks)
- Undertale soundtrack (110 tracks converted to 3207 images, video game music, fair variance between tracks)
- Classical music (95 tracks converted to 16400 images, all of the classical genre, with great variance on composition, based on different composers from different eras and cultures)

For more details refer to the the attached thesis book.

## Data preparation and preprocessing
Utilizing the Python library Mido, we are able to extract information about .mid files. Opening a .mid file with Mido, we are given a list of messages. Each message can tell us whether a note is being pressed ("note_on"), or is let go ("note_off"). Messages also contain information about note pitch (valued in range of 0 to 127), velocity which is the intensity of the sound, and time in ticks. There are also other message types that this work does not utilize, refering to program change, meta messages, etc. 

We convert this list of messages to an image of a whole song in preparation for the convolutions utilized in the neural models. The y axis of the image refers to the note pitch, and the x axis to the time, which is readable even to the human eye. As mentioned time is counted in ticks, and each tick represents a pixel. In practice each track can contain a different type of ticks per beat, so we normalize each track's ticks per beat attribute to 12 for uniformity among images (12 ticks per beat means that a beat is equal to 12 ticks, then a 4/4 measure is 12*4=48 ticks). 
![image](https://user-images.githubusercontent.com/99400979/153410714-3418afc0-9cb1-4047-bae5-e79f574dd51d.png)

Then the whole track image is segmented, creating images with shape 96x96x1. Example of whole song converted to image compared to the same .mid opened in a browser application:
![MIDI_to_image](https://user-images.githubusercontent.com/99400979/153410941-25534d02-aecb-41dc-ad44-14d941aa5266.png)





## GAN
GANs are a powerful tool for generative tasks especially in image generation. A GAN consists of two sub-models, the generator and the discriminator.
The generator receives a vector of random noise as input (in its dense layer) and executes transpose convolutions in order to construct an image. Images produced by the generator are fed to the discriminator along with the ones from the initial dataset. Then the discriminator predicts whether its input is true (from the dataset) or fake (generated). Based on its performance both the submodels are evaluated resulting in a zero-sum game.

### Generator Architecture
![gen_arch](https://user-images.githubusercontent.com/99400979/153491517-b35e61e6-7472-4eb3-8b7b-515e5d6861f5.PNG)

### Discriminator Architecture
![disc_arch](https://user-images.githubusercontent.com/99400979/153491552-2a68a800-d642-440b-ac69-37eb246f0d56.PNG)

### Combined Model:
![gan_combined](https://user-images.githubusercontent.com/99400979/153491593-cafd328f-ff91-4478-8d90-dcc3d184f2a6.PNG)

### Results:

![image](https://user-images.githubusercontent.com/99400979/153493006-91232ead-d59e-4d7f-bf9a-7552d8911adb.png)

Loss & accuracy plots:

![image](https://user-images.githubusercontent.com/99400979/153494975-4a004779-4c81-4cc3-baf0-02594e8f9430.png)
![image](https://user-images.githubusercontent.com/99400979/153495481-87fa9ec8-22c8-497f-b8d7-a3b3cce6a8e8.png)

It is observed that PMD and Undertale iterations suffer from convergence failure (generator loss is increasingly higher from discriminator loss, and discriminator has near perfect accuracy). The Classic iteration though seemingly escapes from that issue, thus outperforming the other two cases. That can be seen based on quality of generated images.

## VAE
VAEs also serve a fair solution for generative modeling. The first submodel, the encoder is fed input images and translates them into encodings. Those encodings then get passed through the second submodel, the decoder, which translates them to images. After training the decoder model is used for image generation based on input encodings that appear in the space. During training, the model is compiled in both reconstruction loss and Kullback-Leibler loss.

### Encoder Architecture:
![encoder_arch](https://user-images.githubusercontent.com/99400979/153566850-2a801684-5a2a-42fd-b0a5-95687dcd0cca.PNG)

### Decoder Architecture:
![decoder_arch](https://user-images.githubusercontent.com/99400979/153566809-e12b6f99-d4d8-4a99-b830-aaba751d7857.PNG)


### Combined Model:
![standard_autoencoder](https://user-images.githubusercontent.com/99400979/153566888-9ab466df-312a-4ea0-a849-fa4618267263.PNG)

### Results:

![image](https://user-images.githubusercontent.com/99400979/153567349-a799204c-cbd4-4f86-84c3-7af3cba591c9.png)

Latent space clustering:
![image](https://user-images.githubusercontent.com/99400979/153567446-a4296119-8789-4724-926f-223786b990f5.png)

Loss plots:

![image](https://user-images.githubusercontent.com/99400979/153567497-827bb6d5-2dcb-4c88-80da-a13479152d5c.png)

It is observed that the Undertale and PMD iterations have a tedency to produce images that resemble musical parts, but in all cases the result is noisier than the GAN approach. Based on loss plots the model behaves as expected.

# Conclusion:
In summary the music produced by those models (GAN and VAE) is not consumer-grade quality, although the models are trained efficiently, with little resource requirements, and behave as expected based on neural network theory. Generative tasks are one of the most difficult problems in machine learning due to the fact that results can be subjective.




