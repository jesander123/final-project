Chromesthesia is a form of synesthesia in which auditory stimuli evoke the perception of color. Current research primarily focuses on the extent to which these sensory systems are activated and how they interact, but places less emphasis on visualizing the specific color associations that synesthetes self-report. Our project aims to make this condition, experienced by only about one percent of the population, more accessible by exploring it via an intersection of art, technology, and cognitive science.

**Chromesthesia Visualizer.py** allows you to input your own audio path to create a unique visualization via PyGame. Audio features including pitch, loudness, and timbre are extracted from your audio input and mapped to corresponding visual features: hue, size, and shape. The music will play synchronously with the video output, simulating a realistic chromesthesia experience. The preassigned colors corresponding to each note are based on a chromesthete’s self-reported experiences (as detailed in Color Data Scraping.ipynb). You can change the RGB values based on your own color associations.

**Feature Extraction.ipynb** provides a more detailed explanation of how each audio feature is extracted and digitized using the Python library librosa, as well as the reasoning behind their selection. Graphs are included to aid in understanding each audio quantification variable (fundamental frequency, root mean square energy, and Mel-frequency cepstral coefficients).

**Audio Features Notes.docx** consolidates information about audio digitization and explains how the functions implemented in Feature Extraction.ipynb work under the hood. This analysis played a key role in our decision regarding which audio descriptors to implement.


