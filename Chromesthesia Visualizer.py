"""
Created on Sun Nov  9 11:56:17 2025

@author: Stella Pan
"""
import librosa
import numpy as np
import librosa.display

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import pygame
import sys
import time
import random

# input any audio file (this is for demonstration purposes)
audio_path = "/Users/stellipan/Desktop/Pic 16B Project/the-best-jazz-club-in-new-orleans-164472.mp3"

# predefined colors for each note (implement user entry later)
NOTE_RGB = {
    "C":  (214, 174, 16),
    "C#": (115, 58, 75),
    "D":  (38, 63, 145),
    "D#": (68, 118, 67),
    "E":  (211, 87, 49),
    "F":  (159, 194, 76),
    "F#": (195, 10, 103),
    "G":  (255, 156, 223),
    "G#": (37, 149, 150),
    "A":  (155, 152, 223),
    "A#": (10, 107, 62),
    "B":  (238, 145, 50)
}
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# predefined cluster shapes 
CLUSTER_SHAPES = {
    0: "diamond", 
    1: "circle",    
    2: "wave",        
    }

def extract_audio_features(audio_path: str, duration: float = None, HOP_LENGTH: int = 2048, FRAME_LENGTH: int = 2048 ):
    """
    This function loads an audio file and extracts audio features: pitch (f0), loudness (RMS energy), and timbre (MFCC) for each beat frame.
    It also normalizes rms and MFCC on a 0-1 scale for visual mapping.
    
    Parameters:
        audio_path: str
            path to any audio file format (.wav, .mp3, .ogg, etc.)
        duration: float, optional
            Duration from the start of the file (seconds). 
            Loads the full track if none. 
        hop_length: int, fixed
            Number of samples between frames.
        frame_length: int, fixed
            Number of samples per frame. 

    Returns:
         audio_features : dict
            times: array of frame timestamps,
            midi: midi number ,
            rms: normalized loudness,
            mfcc: normalized MFCC matrix
    """

    # load audio file
    y, sr = librosa.load(audio_path, duration=duration)

    # 1. Pitch (f0) extraction
    f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                             sr=sr,
                                             frame_length = FRAME_LENGTH,
                                             hop_length = HOP_LENGTH,
                                             fmin=librosa.note_to_hz('A0'),
                                             fmax=librosa.note_to_hz('C8'))
    f0 = np.nan_to_num(f0, nan=np.nanmean(f0))  
    # if there are NaN values, replace them with the mean pitch
    midi = librosa.hz_to_midi(f0)
    # convert frequency to midi note number

    # 2. Loudness(RMS energy) extraction
    rms = librosa.feature.rms(y=y,frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    
    # 3. Timbre (MFCC) extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)

    # time stamps for mapping to pygame 
    times = librosa.times_like(rms, sr=sr, hop_length=HOP_LENGTH)

    # normalize numerical features for mapping (0–1 range)
    def normalize(x):
        min = np.min(x)
        max = np.max(x)
        denom = max - min
        if denom == 0:
            return np.zeros_like(x)  
        return (x - min) / denom
    
    rms_norm = normalize(rms)
    mfcc_norm = np.apply_along_axis(normalize, 1, mfcc)
    
    # K-Means Clustering on MFCC to classify timbre into 3 general groups
    mfcc_T = mfcc_norm.T   # take transpose of MFCC so each row is a time stamp and cols are the 13 features
    scaler = StandardScaler()  
    mfcc_scaled = scaler.fit_transform(mfcc_T) # standardize each coefficent
    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(mfcc_scaled)

    # dictionary of audio features for the funciton to return
    audio_features = {
       "times" : times,
       "midi" : midi,
       "rms" : rms_norm,
       "mfcc" : mfcc_norm,
       "cluster_labels" : cluster_labels
       }
    return audio_features


def coloradjust(rgb, factor):
    """
    This function darken or lighten RGB based on the octaves
    
    Parameters:
        rgb: tuple
            (R,G,B), integers from 0-255
        factor: float
            brightness adjustment factor
            
    returns: 
       newrgb: tuple
           (R,G,B), integers from 0-255
    """
    rgb = np.array(rgb, dtype=float)
    newrgb = rgb * factor
    newrgb = np.clip(newrgb, 0, 255) # any value less than 0 becomes 0, any value more than 255 becomes 255
    return tuple(newrgb.astype(int))

class ChromesthesiaVisualizer:
    """
    Chromesthesia visualization tool that synchronizes PyGame visuals with audio input.
    
    Maps:
        Pitch (midi number) to hue (rgb)
        Loudness (rms) to shape size
        MFCC to shape type
    """

    def __init__(self, audio_path, audio_features, width=1000, height=800, fps=60, fade_time=0.6):
        self.audio_path = audio_path
        self.times = audio_features["times"]
        self.midi = audio_features["midi"]
        self.rms = audio_features["rms"]
        self.mfcc = audio_features["mfcc"]
        self.cluster_labels = audio_features["cluster_labels"]
    
        self.width = width
        self.height = height
        self.fps = fps
        self.fade_time = fade_time

        self.shapes = []
        self.running = False
        self.current_frame = 0

        # initialize pygame window and mixer
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Chromesthesia Visualizer")
        self.clock = pygame.time.Clock()


    def start(self):
        """Start visualization and play audio."""
        
        # load and play audio
        pygame.mixer.music.load(self.audio_path)
        self.start_time = time.time()
        pygame.mixer.music.play()
        self.running = True
        self.current_frame = 0

        # while the visualization is in action
        while self.running and self.current_frame < len(self.times):
            self.stop()
            self.generate()
            self.draw()

            if not pygame.mixer.music.get_busy(): # if the audio is no longer playing
                self.running = False

            pygame.display.flip() # make changes appear on display
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()

    def stop(self):
        """Handle quit events."""
        for event in pygame.event.get(): # for anything event object 
            if event.type == pygame.QUIT: # if event is of type QUIT (eg. close window)
                self.running = False
                pygame.mixer.music.stop() 
                pygame.quit()
                sys.exit()
                
    def generate(self):
        """Generate shapes based on current time stamp."""
        current_time = time.time() - self.start_time  
        
        # While we have not passed the last audio frame and we moved past the last frame
        while (self.current_frame < len(self.times) and current_time >= self.times[self.current_frame]):
            midi = self.midi[self.current_frame]
            rms = self.rms[self.current_frame]
            mfcc = self.mfcc[:, self.current_frame]

            # 1. map hue
            midi_floor = int(np.floor(midi)) # take floor of each midi note
            note_name = NOTE_NAMES[midi_floor % 12] # find their corresponding note name
            base_rgb = NOTE_RGB[note_name] # find the original rbg of each note name
            
            # 2. map brightness : higher octaves have brighter color
            octave = int((midi_floor - 60) // 12) # middle C (C3) is mapped to 0 octave 
            factor = 1.5 + 0.25 * octave # adjust brightness factor depending on octave
            r, g, b = coloradjust(base_rgb, factor)

            # 3. map size : louder sound have larger size 
            radius = int(10 + rms * 80) 
        
            # 4. map smoothness of sound to fade time or shapes: smooth sounds fade slower
            hcoeff= mfcc[6:] # if sound is smooth, the values of higher-order coefficents will be close to 0
            energy = np.mean(np.abs(hcoeff)) # get mean of the absolute value of higher-order coefficents
            smoothness = 1.0 / (1.0 + energy) # more energy = less smooth
            self.fade_time = 0.3 + smoothness * 0.8 

            # 5. map mfcc to shape: percussive instruments = circles, string instruments = waves, sharp sounds = diamonds
            cluster = self.cluster_labels[self.current_frame]
            shape_type = CLUSTER_SHAPES[cluster]

            # 6. map distribution of the shapes
            if shape_type == "circle":
                # bottom left
                x = random.randint(int(self.width * 0.15), int(self.width * 0.55))
                y = random.randint(int(self.height * 0.50), int(self.height * 0.90))

            elif shape_type == "diamond":
                # upper right
                x = random.randint(int(self.width * 0.50), int(self.width * 0.90))
                y = random.randint(int(self.height * 0.15), int(self.height * 0.50))

            elif shape_type == "wave":
                # center
                x = random.randint(int(self.width*0.45), int(self.width*0.55))
                y = random.randint(int(self.height*0.30), int(self.height*0.45))
                
            self.shapes.append({
                "x": x,
                "y": y,
                "radius": radius,
                "color": (r, g, b),
                "alpha": 255,
                "type": shape_type,
                "spawn_time": current_time
            })
            self.current_frame += 1

        # fade out shapes, remove when alpha = 0
        self.shapes = [s for s in self.shapes if self.fade(s, current_time)]

    def fade(self, shape, current_time):
        """Reduce alpha until transparent, then return false"""
        elaspe = current_time - shape["spawn_time"]
        shape["alpha"] = max(0, 255 * (1 - elaspe / self.fade_time))# start opaque, fade according to fade time
        return shape["alpha"] > 0


    def draw(self):
        """draw the shapes on the window"""
        self.screen.fill((0, 0, 0)) # black background

        for s in self.shapes:
            radius = int(s["radius"]) # size of shape
            color = (*s["color"], int(s["alpha"]))# unpacks the color tuple and adds alpha to it
            
            if s["type"] in ("circle", "diamond"):
                surf_size = radius * 4 # size of surface
                surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)# surface for shapes to appear on, SRCALPHA allows transprency 
                cx, cy = surf_size // 2, surf_size // 2 # x and y coords for the center of the surface

                if s["type"] == "circle":
                    pygame.draw.circle(surf, color, (cx, cy), radius)

                elif s["type"] == "diamond":
                    height = int(40 + radius * 2) # louder = longer diamond
                    width = max(3, int(radius * 0.5)) # louder = wider diamond

                    # stack horizontal lines to make a diamond shape
                    for i in range(height): # moves from top to bottom
                        offset = i - height // 2  # guarantees symmetry: negative = top half, positive = bottom half
                    
                        # taper towards end
                        t = 1 - abs(offset) / (height // 2)
                        line_length = max(1, int(width * t)) # convert t to number of pixels (never smaller than 1)
                    
                        #find row
                        y = cy + offset
                    
                        pygame.draw.line(
                            surf,
                            color,
                            (cx - line_length// 2, y), # from left
                            (cx + line_length // 2, y), # to right
                            1 # each line is 1 pixel thick
                            )
                        
                surf = surf.convert_alpha()    # prep surface for gaussian blur
                blurred = pygame.transform.gaussian_blur(surf, radius=9)  
                self.screen.blit(blurred, (s["x"] - cx, s["y"] - cy))
                # blit copies one surface onto another, replace surf with the window screen
                # correct for the center of the shape

            elif s["type"] == "wave":
                wave_width = self.width # width is the same as the entire screen
                wave_height = int(self.height * 0.25) # height is 1/4 the screen, give enough room for sin wave but still efficent 
                wave_surf = pygame.Surface((wave_width, wave_height), pygame.SRCALPHA) # local scurface to draw on
                
                cy = wave_height // 2 # local center 
                
                wave_length = 50 + radius * 2 # louder = longer wavelength
                amplitude = max(10, radius) # louder = taller
                thickness = max(2, radius // 8) # louder = thicker

                points = [] # store points that form the wace
                for x in range(wave_width): # for every x 
                    gy = int(s["y"] + amplitude * np.sin(2*np.pi * x / wave_length)) # use global (screen) coords to compute sin wave 
                    y = gy - (s["y"] - cy) # convert world coords to local
                    points.append((x, y))

                pygame.draw.lines(
                    wave_surf, 
                    color,
                    False, # do no close the shape, only show a line
                    points, 
                    thickness)
                
                wave_surf = wave_surf.convert_alpha() # prep surface for gaussian blur
                blurred_wave = pygame.transform.gaussian_blur(wave_surf, radius=4)
                self.screen.blit(blurred_wave, (0, s["y"] - cy))
                # blit copies one surface onto another, replace wave_surf with the window screen
                # correct for the center of the shape

features = extract_audio_features(audio_path)
visualizer = ChromesthesiaVisualizer(audio_path, features)
visualizer.start()
