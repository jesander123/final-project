Chromesthesia is a form of synesthesia in which auditory stimuli evoke the perception of color. Current research primarily focuses on the extent to which these sensory systems are activated and how they interact, but places less emphasis on visualizing the specific color associations that synesthetes self-report. Our project aims to make this condition, experienced by only about one percent of the population, more accessible by exploring it via an intersection of art, technology, and cognitive science. The order in which the files are listed below is the order in which you should run them to simulate the web application format of visualization output. 

**Color Data Scraping.ipynb** documents the process of webscraping the “Synesthesia Tree” website to obtain note-to-color associations. This document also includes functions to adjust the color of the “base note” in reference to its octave. Lastly, it includes a static image visualization of a Chromatic Circle, representing the relationships between the 12 notes of the chromatic scale and their corresponding colors that synesthetes experience. 

**Feature Extraction.ipynb** provides a more detailed explanation of how each audio feature is extracted and digitized using the Python library librosa, as well as the reasoning behind their selection. Graphs are included to aid in understanding each audio quantification variable (fundamental frequency, root mean square energy, and Mel-frequency cepstral coefficients).

**Audio Features Notes.docx** consolidates information about audio digitization and explains how the functions implemented in Feature Extraction.ipynb work under the hood. This analysis played a key role in our decision regarding which audio descriptors to implement.

**Neural Network Visualizer.ipynb** constructs a simplified neural network architecture; it consists of input, association, and output layers that are meant to mimic the propagation of an active potential through neurons in a human neural network. The input layer consists of twelve nodes associated with notes (C, C#, D… B), the association layer represents the octave factor, and the output layer is the visualization that represents the color-to-note associations as selected by the user (these nodes also pulse in accordance with the note’s loudness). The network is synchronized with the audio and the nodes pulse in time with the frame in which the auditory feature associated with the node is present. 

**Chromesthesia Visualizer.py** allows users to input their own audio path to create a unique visualization via PyGame. Audio features including pitch, loudness, and timbre are extracted from your audio input and mapped to corresponding visual features: hue, size, and shape. The music will play synchronously with the video output, simulating a realistic chromesthesia experience. The preassigned colors corresponding to each note are based on a chromesthete’s self-reported experiences (as detailed in Color Data Scraping.ipynb). You can change the RGB values based on your own color associations.






