import mido
from mido import MidiFile
import numpy as np

def write_midi(pr, ticks_per_beat, write_path, tempo=80):
    def pr_to_list(pr):
        # List event = (pitch, velocity, time)
        T, N = pr.shape
        t_last = 0
        pr_tm1 = np.zeros(N)
        list_event = []
        for t in range(T):
            pr_t = pr[t]
            mask = (pr_t != pr_tm1)
            if (mask).any():
                for n in range(N):
                    if mask[n]:
                        pitch = n
                        velocity = int(pr_t[n])
                        # Time is incremented since last event
                        t_event = t - t_last
                        t_last = t
                        list_event.append((pitch, velocity, t_event))
            pr_tm1 = pr_t
        return list_event
    # Tempo
    microseconds_per_beat = mido.bpm2tempo(tempo)
    # Write a pianoroll in a midi file
    mid = MidiFile()
    mid.ticks_per_beat = ticks_per_beat

    # Each instrument is a track
    for instrument_name, matrix in pr.items():
        # Add a new track with the instrument name to the midi file
        track = mid.add_track(instrument_name)
        # transform the matrix in a list of (pitch, velocity, time)
        events = pr_to_list(matrix)
        # Tempo
        track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
        # Add the program_change
        try:
            program = program_change_mapping[instrument_name]
        except:
            # Defaul is piano
            # print instrument_name + " not in the program_change mapping"
            # print "Default value is 1 (piano)"
            # print "Check acidano/data_processing/utils/program_change_mapping.py"
            program = 1
        track.append(mido.Message('program_change', program=program))

        # This list is required to shut down
        # notes that are on, intensity modified, then off only 1 time
        # Example :
        # (60,20,0)
        # (60,40,10)
        # (60,0,15)
        notes_on_list = []
        # Write events in the midi file
        for event in events:
            pitch, velocity, time = event
            if velocity == 0:
                # Get the channel
                track.append(mido.Message('note_off', note=pitch, velocity=0, time=time))
                notes_on_list.remove(pitch)
            else:
                if pitch in notes_on_list:
                    track.append(mido.Message('note_off', note=pitch, velocity=0, time=time))
                    notes_on_list.remove(pitch)
                    time = 0
                track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=time))
                notes_on_list.append(pitch)
    mid.save(write_path)
    return

#!/usr/bin/env python
# -*- coding: utf8 -*-

# from mido import MidiFile
from unidecode import unidecode
# import numpy as np

#######
# Pianorolls dims are  :   TIME  *  PITCH


class Read_midi(object):
    def __init__(self, song_path, quantization):
        ## Metadata
        self.__song_path = song_path
        self.__quantization = quantization

        ## Pianoroll
        self.__T_pr = None

        ## Private misc
        self.__num_ticks = None
        self.__T_file = None

    @property
    def quantization(self):
        return self.__quantization

    @property
    def T_pr(self):
        return self.__T_pr

    @property
    def T_file(self):
        return self.__T_file

    def get_total_num_tick(self):
        # Midi length should be written in a meta message at the beginning of the file,
        # but in many cases, lazy motherfuckers didn't write it...

        # Read a midi file and return a dictionnary {track_name : pianoroll}
        mid = MidiFile(self.__song_path)

        # Parse track by track
        num_ticks = 0
        for i, track in enumerate(mid.tracks):
            tick_counter = 0
            for message in track:
                # Note on
                time = float(message.time)
                tick_counter += time
            num_ticks = max(num_ticks, tick_counter)
        self.__num_ticks = num_ticks

    def get_pitch_range(self):
        mid = MidiFile(self.__song_path)
        min_pitch = 200
        max_pitch = 0
        for i, track in enumerate(mid.tracks):
            for message in track:
                if message.type in ['note_on', 'note_off']:
                    pitch = message.note
                    if pitch > max_pitch:
                        max_pitch = pitch
                    if pitch < min_pitch:
                        min_pitch = pitch
        return min_pitch, max_pitch

    def get_time_file(self):
        # Get the time dimension for a pianoroll given a certain quantization
        mid = MidiFile(self.__song_path)
        # Tick per beat
        ticks_per_beat = mid.ticks_per_beat
        # Total number of ticks
        self.get_total_num_tick()
        # Dimensions of the pianoroll for each track
        self.__T_file = int((self.__num_ticks / ticks_per_beat) * self.__quantization)
        return self.__T_file

    def read_file(self):
        # Read the midi file and return a dictionnary {track_name : pianoroll}
        mid = MidiFile(self.__song_path)
        # Tick per beat
        ticks_per_beat = mid.ticks_per_beat

        # Get total time
        self.get_time_file()
        T_pr = self.__T_file
        # Pitch dimension
        N_pr = 128
        pianoroll = {}

        def add_note_to_pr(note_off, notes_on, pr):
            pitch_off, _, time_off = note_off
            # Note off : search for the note in the list of note on,
            # get the start and end time
            # write it in th pr
            match_list = [(ind, item) for (ind, item) in enumerate(notes_on) if item[0] == pitch_off]
            if len(match_list) == 0:
                print("Try to note off a note that has never been turned on")
                # Do nothing
                return

            # Add note to the pr
            pitch, velocity, time_on = match_list[0][1]
            pr[time_on:time_off, pitch] = velocity
            # Remove the note from notes_on
            ind_match = match_list[0][0]
            del notes_on[ind_match]
            return

        # Parse track by track
        counter_unnamed_track = 0
        for i, track in enumerate(mid.tracks):
            # Instanciate the pianoroll
            pr = np.zeros([T_pr, N_pr])
            time_counter = 0
            notes_on = []
            for message in track:

                ##########################################
                ##########################################
                ##########################################
                # TODO : keep track of tempo information
                # import re
                # if re.search("tempo", message.type):
                #     import pdb; pdb.set_trace()
                ##########################################
                ##########################################
                ##########################################


                # print message
                # Time. Must be incremented, whether it is a note on/off or not
                time = float(message.time)
                time_counter += time / ticks_per_beat * self.__quantization
                # Time in pr (mapping)
                time_pr = int(round(time_counter))
                # Note on
                if message.type == 'note_on':
                    # Get pitch
                    pitch = message.note
                    # Get velocity
                    velocity = message.velocity
                    if velocity > 0:
                        notes_on.append((pitch, velocity, time_pr))
                    elif velocity == 0:
                        add_note_to_pr((pitch, velocity, time_pr), notes_on, pr)
                # Note off
                elif message.type == 'note_off':
                    pitch = message.note
                    velocity = message.velocity
                    add_note_to_pr((pitch, velocity, time_pr), notes_on, pr)

            # We deal with discrete values ranged between 0 and 127
            #     -> convert to int
            pr = pr.astype(np.int16)
            if np.sum(np.sum(pr)) > 0:
                name = unidecode(track.name)
                name = name.rstrip('\x00')
                if name == u'':
                    name = 'unnamed' + str(counter_unnamed_track)
                    counter_unnamed_track += 1
                if name in pianoroll.keys():
                    # Take max of the to pianorolls
                    pianoroll[name] = np.maximum(pr, pianoroll[name])
                else:
                    pianoroll[name] = pr
        return pianoroll



def get_pianoroll_time(pianoroll):
    T_pr_list = []
    for k, v in pianoroll.items():
        T_pr_list.append(v.shape[0])
    if not len(set(T_pr_list)) == 1:
        print("Inconsistent dimensions in the new PR")
        return None
    return T_pr_list[0]

def get_pitch_dim(pianoroll):
    N_pr_list = []
    for k, v in pianoroll.items():
        N_pr_list.append(v.shape[1])
    if not len(set(N_pr_list)) == 1:
        print("Inconsistent dimensions in the new PR")
        raise NameError("Pr dimension")
    return N_pr_list[0]

def dict_to_matrix(pianoroll):
    T_pr = get_pianoroll_time(pianoroll)
    N_pr = get_pitch_dim(pianoroll)
    rp = np.zeros((T_pr, N_pr), dtype=np.int16)
    for k, v in pianoroll.items():
        if rp.sum() < v.sum():
            rp = v
    return rp


import numpy as np
import pandas as pd
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# from torchvision.transforms import Compose, ToTensor, Lambda

def visualise_op(images, title=""):
    """Shows the provided images as sub-pictures in a square"""
    print(images.shape)
    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], norm=None)
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()



def generate_new_samples(ddpm, n_samples=4, device=None, c=1, h=128, w=128):
    """Given a DDPM model, a number of samples to be generated and a device, 
        returns some newly generated samples"""

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                x = x + sigma_t * z
    return x


def sinusoidal_embedding(n, d):
   # Returns the standard positional embedding
   embedding = torch.zeros(n, d)
   wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
   wk = wk.reshape((1, d))
   t = torch.arange(n).reshape((n, 1))
   embedding[:,::2] = torch.sin(t * wk[:,::2])
   embedding[:,1::2] = torch.cos(t * wk[:,::2])

   return embedding

class SelfAttention(nn.Module):
   def __init__(self, channels):
       super(SelfAttention, self).__init__()
       self.channels = channels        
       num_heads = 2 if channels == 10 else 4 
       self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
       self.ln = nn.LayerNorm([channels])
       self.ff_self = nn.Sequential(
           nn.LayerNorm([channels]),
           nn.Linear(channels, channels),
           nn.GELU(),
           nn.Linear(channels, channels),
       )

   def forward(self, x):
       size = x.shape[-1]
       x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
       x_ln = self.ln(x)
       attention_value, _ = self.mha(x_ln, x_ln, x_ln)
       attention_value = attention_value + x
       attention_value = self.ff_self(attention_value) + attention_value
       return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)

class Empty(nn.Module):
   def __init__(self):
       super(Empty, self).__init__()
       
   def forward(self, x):
       return x
   
# DDPM class
class DDPMModel(nn.Module):
   def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 192, 128)):
       super(DDPMModel, self).__init__()
       self.n_steps = n_steps
       self.device = device
       self.image_chw = image_chw
       self.network = network.to(device)
       self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # Number of steps is typically in the order of thousands
#         self.betas = cosine_beta_schedule(n_steps).to(device)
       self.alphas = 1 - self.betas
       self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

   def forward(self, x0, t, eta=None):
       # Make input image more noisy (we can directly skip to the desired step)
       n, c, h, w = x0.shape
       a_bar = self.alpha_bars[t]

       if eta is None:
           eta = torch.randn(n, c, h, w).to(self.device)

       noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
       return noisy

   def backward(self, x, t):
       # Run each image through the network for each timestep t in the vector t.
       # The network returns its estimation of the noise that was added.
       return self.network(x, t)
   
class DConvBlock(nn.Module):
   def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
       super(DConvBlock, self).__init__()
       self.s_shape = shape
       self.in_c = in_c
       self.out_c = out_c
       self.ln = nn.LayerNorm(shape)
       self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
       self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
       self.activation = nn.SiLU() if activation is None else activation
       self.normalize = normalize

   def forward(self, x):
       out = self.ln(x) if self.normalize else x
       out = self.conv1(out)
       out = self.activation(out)
       out = self.conv2(out)
       out = self.activation(out)
       return out
   
class MyUNet(nn.Module):
   def __init__(self, n_steps, time_emb_dim=100):
       super(MyUNet, self).__init__()

       ################################################################# Sinusoidal embedding
       self.time_embed = nn.Embedding(n_steps, time_emb_dim)
       self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
       self.time_embed.requires_grad_(False)

       ################################################################ First half
       self.te1 = self._make_te(time_emb_dim, 1)
       self.b1 = nn.Sequential(
           DConvBlock((1, 192, 128), 1, 10, kernel_size=5, padding=2),
           DConvBlock((10, 192, 128), 10, 20, kernel_size=5, padding=2),
           DConvBlock((20, 192, 128), 20, 20, kernel_size=5, padding=2)
       )
       self.down1 = nn.Conv2d(20, 20, (6,4), (3,2), (2,1))
       
       self.te11 = self._make_te(time_emb_dim, 20)
       self.b11 = nn.Sequential(
           DConvBlock((20, 64, 64), 20, 20),
           DConvBlock((20, 64, 64), 20, 20),
           DConvBlock((20, 64, 64), 20, 20)
       )
       self.down11 = nn.Conv2d(20, 20, 4, 2, 1)
       
       self.te12 = self._make_te(time_emb_dim, 20)
       self.b12 = nn.Sequential(
           DConvBlock((20, 32, 32), 20, 20),
           SelfAttention(20),
           DConvBlock((20, 32, 32), 20, 20),
           DConvBlock((20, 32, 32), 20, 20)
       )
       self.down12 = nn.Conv2d(20, 20, 4, 2, 1)
       
       self.te2 = self._make_te(time_emb_dim, 20)
       self.b2 = nn.Sequential(
           DConvBlock((20, 16, 16), 20, 20),
           SelfAttention(20),
           DConvBlock((20, 16, 16), 20, 20),
           DConvBlock((20, 16, 16), 20, 20)
       )
       self.down2 = nn.Conv2d(20, 20, 4, 2, 1)
       
       self.te3 = self._make_te(time_emb_dim, 20)
       self.b3 = nn.Sequential(
           DConvBlock((20, 8, 8), 20, 40),
           SelfAttention(40),
           DConvBlock((40, 8, 8), 40, 40),
           DConvBlock((40, 8, 8), 40, 40)
       )
       self.down3 = nn.Sequential(
           nn.Conv2d(40, 40, 2, 1),
           nn.SiLU(),
           nn.Conv2d(40, 40, 4, 2, 1)
       )

       ######################################################## Bottleneck
       self.te_mid = self._make_te(time_emb_dim, 40)
       self.b_mid = nn.Sequential(
           DConvBlock((40, 3, 3), 40, 20),
           SelfAttention(20),
           DConvBlock((20, 3, 3), 20, 20),
           DConvBlock((20, 3, 3), 20, 40)
       )

       ######################################################## Second half
       self.up1 = nn.Sequential(
           nn.ConvTranspose2d(40, 40, 3, 3, 1),
           nn.SiLU(),
           nn.ConvTranspose2d(40, 40, 2, 1)
       )

       self.te4 = self._make_te(time_emb_dim, 80)
       self.b4 = nn.Sequential(
           DConvBlock((80, 8, 8), 80, 40),
           SelfAttention(40),
           DConvBlock((40, 8, 8), 40, 20),
           DConvBlock((20, 8, 8), 20, 20)
       )

       self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
       
       self.te5 = self._make_te(time_emb_dim, 40)
       self.b5 = nn.Sequential(
           DConvBlock((40, 16, 16), 40, 20),
           SelfAttention(20),
           DConvBlock((20, 16, 16), 20, 20),
           DConvBlock((20, 16, 16), 20, 20)
       )

       self.up21 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
       self.te51 = self._make_te(time_emb_dim, 40)
       self.b51 = nn.Sequential(
           DConvBlock((40, 32, 32), 40, 40),
           SelfAttention(40),
           DConvBlock((40, 32, 32), 40, 20),
           DConvBlock((20, 32, 32), 20, 20)
       )

       self.up22 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
       self.te52 = self._make_te(time_emb_dim, 40)
       self.b52 = nn.Sequential(
           DConvBlock((40, 64, 64), 40, 40),
           DConvBlock((40, 64, 64), 40, 20),
           DConvBlock((20, 64, 64), 20, 20)
       )

       self.up3 = nn.ConvTranspose2d(20, 20, (7,4), (3,2), (2,1))
       self.te_out = self._make_te(time_emb_dim, 40)
       self.b_out = nn.Sequential(
           DConvBlock((40, 192, 128), 40, 20),
           DConvBlock((20, 192, 128), 20, 10),
           DConvBlock((10, 192, 128), 10, 10, normalize=False)
       )
       
       self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

   def forward(self, x, t):
       # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
           
       
       t = self.time_embed(t)
       n = len(x)
       out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))                          # (N, 10, 128, 128)
#         print(out1.shape)
#         out1 = self.sa1(out1)
       out11 = self.b11(self.down1(out1) + self.te11(t).reshape(n, -1, 1, 1))        # (N, 10, 64, 64)
#         out11 = self.sa11(out11)
       out12 = self.b12(self.down11(out11) + self.te12(t).reshape(n, -1, 1, 1))      # (N, 10, 32, 32)
#         out12 = self.sa12(out12)
       out2 = self.b2(self.down12(out12) + self.te2(t).reshape(n, -1, 1, 1))         # (N, 20, 16, 16)
#         out2 = self.sa2(out2)
       out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))           # (N, 40, 8, 8)
#         out3 = self.sa3(out3)

       out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

       out4 = torch.cat((out3, self.up1(out_mid)), dim=1)                            # (N, 80, 8, 8)
       out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))                       # (N, 20, 8, 8)
#         out4 = self.sa4(out4)

       out5 = torch.cat((out2, self.up2(out4)), dim=1)                               # (N, 40, 16, 16)
       out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))                       # (N, 10, 16, 16)
#         out5 = self.sa5(out5)

       out51 = torch.cat((out12, self.up21(out5)), dim=1)                            # (N, 20, 32, 32)
       out51 = self.b51(out51 + self.te51(t).reshape(n, -1, 1, 1))                   # (N, 10, 32, 32)
#         out51 = self.sa51(out51)

       out52 = torch.cat((out11, self.up22(out51)), dim=1)                           # (N, 40, 64, 64)
       out52 = self.b52(out52 + self.te52(t).reshape(n, -1, 1, 1))                  # (N, 10, 64, 64)
#         out52 = self.sa52(out52)
#         print(self.up3(out52).shape)
       out = torch.cat((out1, self.up3(out52)), dim=1)                               # (N, 20, 128, 128)
       out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))                   # (N, 1, 128, 128)
#         out = self.saout(out)

       out = self.conv_out(out)

       return out

   def _make_te(self, dim_in, dim_out):
       return nn.Sequential(
           nn.Linear(dim_in, dim_out),
           nn.SiLU(),
           nn.Linear(dim_out, dim_out)
       )

device = 'cuda'

def hp_filter(a, mean):
    if a < mean*7:
        return 0
    return a

def create_midi(audio, channels=1):
    audio = audio * (127/audio.max())
    audio = audio - audio.min()
    audio = audio * (127/audio.max())

    audio = audio.reshape(-1)

    audio = np.array([hp_filter(a, audio.mean()) for a in audio])
    
    audio = audio * (127/audio.max())

    audio = audio.reshape(1,channels,-1,128)
    return audio #audio.astype(int)

def midi_post_process(audio, r=192, c=128):
    print(audio.shape)
    audio = audio.reshape(r,c).T
    for i in range(c):
        for j in range(r):
            if abs(audio[i][j] - audio[i][j-1]) < 127*0.1 and audio[i][j] > 0:
                audio[i][j] = audio[i][j-1]
        
    audio = audio.T
    audio = audio.reshape(1,1,r,c)
    return audio



def save_samples(model, loc):
    print("-----------------")
    print(loc)
    print("-----------------")
    generated = generate_new_samples(
        best_model,
        n_samples=5,
        device=device,
        h=height
    )
    
    # visualise_op(generated)    
    
    print('Generation complete')
    for i,audio in enumerate(generated):
        try:
            audio = audio.to('cpu').numpy().reshape(192,128)
            audio = create_midi(audio, channels=1)

            audio = midi_post_process(audio)

            # visualise_op(audio)

            # df1 = pd.DataFrame(audio.reshape(192,128))

            save_loc = loc + "/sample_" + str(i) + ".mid"

            write_midi({'Track1': audio.reshape(192,128).astype(int)}, 4, save_loc, 122)
        except:
            pass


weight_path = 'ddpm_model_q3_with_attention_17Aug.pt'    
    
n_steps=1000
height = 192

print("GPU -->", torch.cuda.is_available())

folder_name = "Sample"

best_model = DDPMModel(MyUNet(n_steps), n_steps=n_steps, device=device)
best_model.load_state_dict(torch.load(weight_path, map_location=device))
best_model.eval()
print()
#             for i in range (20):
save_samples(best_model, folder_name)