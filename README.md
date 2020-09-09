# LSTM-GAN-Prototype
This is a prototype model for exoplanet transit curve generation. The generator uses a Fully connected layer which takes in a random sequence of data and outputs a lightcurve-like graph when plotted. The discriminator consists of LSTM layers stacked on top of each other which uses the labelled Kaggle data and outputs 1 or -1 depending on whether the lightcurve represents an exoplanet or not. This is only a prototype model for my Masters research project, in the future I plan to improve the model architecture to generate more realistic "exoplanets". The ultimate goal of this model is to generate exoplanets which look real to us but it could potentially fool a ML model.

The data from Kaggle can be downloaded from this link: https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data

As a reference, this is a real transit curve for an exoplanet. 
,
,
,
,
<img src="https://github.com/peterfazekas1999/LSTM-GAN-Prototype/blob/master/GAN-LSTM%20generated/GAN_3_real.jpg" width ="200">
<img src="https://github.com/peterfazekas1999/LSTM-GAN-Prototype/blob/master/GAN-LSTM%20generated/real_curve.png" width ="200">

And these are a few transit curves generated by the model.
<img src="https://github.com/peterfazekas1999/LSTM-GAN-Prototype/blob/master/GAN-LSTM%20generated/GAN_11.png" width ="200">
<img src="https://github.com/peterfazekas1999/LSTM-GAN-Prototype/blob/master/GAN-LSTM%20generated/GAN_10.png" width ="200">


