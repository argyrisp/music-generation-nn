import mido     # https://mido.readthedocs.io/en/latest/
import numpy as np
import cv2
import os

time_dif = 12
time_of_image = 96
total_songs = 0
dataset_selection = "undertale"


def dir_recursion(path):
    global total_songs
    global img_id
    global save_dir
    entries = os.listdir(path + '/')
    for item in entries:
        if os.path.isdir(path+'/'+item):
            dir_recursion(path+'/'+item)
            # recursion if its folder
        elif os.path.isfile(path+'/'+item):
            # print(item)
            if item.endswith(".mid"):
                print(item)
                fname = path + "\\" + item
                image = open_messages(fname)
                total_time = len(image[0])
                # Uncomment bellow to save the whole song as .png in current path.
                # cv2.imwrite(str(item).split('.')[0] + '.png', image)
                save_dataset(image, total_time)
                total_songs = total_songs+1
        else:
            print("Warning! - dir_recursion bug.")


def save_dataset(image, total_time):
    global img_id
    global save_dir
    offset = 0

    while offset + (time_of_image-1) < total_time:
        temp_img = np.zeros((96, time_of_image))
        empty = True
        for y in range(offset, offset + time_of_image):
            for x in range(0, 96):
                temp_img[x][y - offset] = image[x][y]
                if image[x][y] != 0.0:
                    empty = False
        offset += time_of_image
        if empty:
            continue
        cv2.imwrite(save_dir+str(img_id)+'.png', temp_img)
        # print('image number ', img_id, " saved with range: ", offset, " to ", offset + 95)
        img_id += 1


def concatenate_tracks(image, track_image):
    for x in range(len(track_image)):
        for y in range(len(track_image[0])):
            image[x][y] = max(image[x][y], track_image[x][y])
    return image


def create_image(total_time, messages):
    image = np.zeros((96, total_time + time_dif * 8))  # + time_dif * 4))
    curr_time = 0
    k = 0
    notes = []
    notes_dict = {}
    terminate = False
    for msg in messages:
        # print(msg)
        k += msg['time']
        if terminate:
            break
        curr_time = msg['time']

        for curr_notes in notes_dict.keys():
            if terminate:
                break
            for time in range(k - curr_time, k):
                if k - curr_time > total_time:
                    terminate = True
                    break
                # image[96 - int(curr_notes)][time] = notes_dict[str(curr_notes)]
                if notes_dict[str(curr_notes)] == 'on':     # indicates note on
                    image[96 - int(curr_notes)][time] = 255.0
                if notes_dict[str(curr_notes)] == 'hold':
                    image[96 - int(curr_notes)][time] = 127.0

                # image[int(curr_notes)][time] = notes_dict[str(curr_notes)]
                if notes_dict[str(curr_notes)] == 'on':        # == 255.0
                    notes_dict[str(curr_notes)] = 'hold'        # == 127.0
        if msg['type'] == 'note_on' and msg['velocity'] != 0:
            notes_dict[str(msg['note'])] = 'on'
        if msg['type'] == 'note_off':
            if str(msg['note']) in notes_dict:
                del notes_dict[str(msg['note'])]
    return image


def open_messages(fname):
    mid = mido.MidiFile(fname)
    track_list = mido.MidiTrack()
    track_list.append(mido.MetaMessage(type='set_tempo', tempo=600000, time=0))
    image = np.zeros((96, 0))

    # print("Ticks per beat: ", mid.ticks_per_beat)
    for i, track in enumerate(mid.tracks):
        total_time = 0
        dt = 0
        switch = True
        messages = []
        # print('Track {}: {}'.format(i, track.name))
        for msg in track:
            # print(msg)
            if msg.type == 'note_on' or msg.type == 'note_off':
                if switch:
                    msg.time = 0
                    dt = 0
                    total_time = 0
                    switch = False
                if msg.velocity == 0 and msg.type == 'note_on':
                    msg.time = msg.time + dt
                    messages.append({'type': 'note_off', 'note': msg.note, 'velocity': msg.velocity,
                                     'time': round(msg.time * time_dif / mid.ticks_per_beat)})
                    track_list.append(mido.Message(type='note_off', note=msg.note, velocity=msg.velocity,
                                                   time=round(msg.time * time_dif / mid.ticks_per_beat)))
                    dt = 0
                else:
                    msg.time = msg.time + dt
                    messages.append({'type': msg.type, 'note': msg.note, 'velocity': msg.velocity,
                                     'time': round(msg.time * time_dif / mid.ticks_per_beat)})
                    track_list.append(mido.Message(msg.type, note=msg.note, velocity=msg.velocity,
                                                   time=round(msg.time * time_dif / mid.ticks_per_beat)))
                    dt = 0
            else:
                dt += msg.time
            total_time += msg.time
        total_time = round(total_time * time_dif / mid.ticks_per_beat)
        # function calls
        track_image = create_image(total_time, messages)    # create an image of notes for the track
        # cv2.imwrite('whole_song'+str(i)+'.png', track_image)
        if len(image[0]) > len(track_image[0]):
            image = concatenate_tracks(image, track_image)
        else:
            image = concatenate_tracks(track_image, image)
        # image = concatenate_tracks(image, track_image)      # stitch together tracks to one image
        # print(np.shape(image))
    # total_time = round(total_time * 12 / mid.ticks_per_beat)

    # From 1st "note press" as 255 and rest pixels that are "note held" ---> whole note press and held as 255
    # kept like this because in a previous approach, multiple pixel coloring was thought to help
    for y in range(len(image[0])):
        for x in range(len(image)):
            if image[x][y] == 127.0:
                if image[x][y + 1] == 255.0:
                    image[x][y] = 0.0
                else:
                    image[x][y] = 255.0
    # fill the notes
    '''    for y in range(len(image[0])):
        for x in range(len(image)):
            if image[x][y] == 255.0:
                if y > 0 and image[x][y - 1] == 0.0:
                    image[x][y - 1] = 127.0
                if x > 0 and image[x - 1][y] == 0.0:
                    image[x - 1][y] = 127.0
                if x < len(image)-1-1:
                    if image[x + 1][y] == 0.0:
                        image[x + 1][y] = 127.0
                if y < len(image[0])-1-1:
                    if image[x][y + 1] == 0.0:
                        image[x][y + 1] = 127.0'''
    return image


whole_seq = []

path = 'MIDIs\\' + dataset_selection
entries = os.listdir(path+'/')

if not os.path.exists("datasets"):
    os.mkdir("datasets")

save_dir = 'datasets\\' + dataset_selection + "\\"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

img_id = 0
dir_recursion(path)
print("MIDI to image conversion for the "+dataset_selection+" MIDI files complete.")
print("Total songs: ", total_songs)
print("Total images: ", img_id)
ftxt = open("datasets\\" + dataset_selection + ".txt", "w")
ftxt.write("Total songs: " + str(total_songs) + "\n")
ftxt.write("Total images: " + str(img_id) + "\n")
ftxt.close()


'''classical_dataset_switch = False
if classical_dataset_switch:
    for folder in entries:
        print(folder)
        files = os.listdir(path+'/'+folder+'/')
        for file in files:
            print(file)
            fname = path+"\\"+folder+"\\"+file
            image = open_messages(fname)
            total_time = len(image[0])
            cv2.imwrite('full_song_dir\\'+str(file)+'.png', image)
            img_id = save_dataset(save_dir, img_id, image, total_time)
else:
    for filename in os.listdir(path):
        if filename.endswith(".mid"):
            print(filename)
            fname = path + "\\" + filename
            image = open_messages(fname)
            total_time = len(image[0])
            # cv2.imwrite('full_song_dir\\' + str(filename) + '.png', image)
            # img_id = save_dataset(save_dir, img_id, image, total_time)
            break'''

