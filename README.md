# LSTM-GAN-Prototype
This is a prototype model for exoplanet transit curve generation. The generator uses a Fully connected layer which takes in a random sequence of data and outputs a lightcurve-like graph when plotted. The discriminator consists of LSTM layers stacked on top of each other which uses the labelled Kaggle data as well as the output of the Generator and classifies the generated lightcurve during training depending on whether it represents an exoplanet or not. This is only a prototype model for my Masters research project, in the future I plan to improve the model architecture to generate more realistic "exoplanets". The ultimate goal of this model is to generate exoplanets which look real to us but it could potentially fool a ML model.

The Generator seems to capture the noise characteristics of the data really well, but not the periodic dips that occurs during the transit of a planet in front of the host star. A possible improvement to the model could be to use an existing python package to artificially insert these "dips" on a GAN generated image and then test how a pre-existing classifier reacts to these artificially crafted planets.

The data from Kaggle can be downloaded from this link: https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data

As a reference, these are real transit curve for an exoplanet. <br />
<img src="https://github.com/peterfazekas1999/LSTM-GAN-Prototype/blob/master/GAN-LSTM%20generated/GAN_3_real.jpg" width ="300">
<img src="https://github.com/peterfazekas1999/LSTM-GAN-Prototype/blob/master/GAN-LSTM%20generated/real_curve.png" width ="300">

The first image is the output of the GAN, the second image is the same output after artificially inserting exoplanet transits.<br />
<img src="https://github.com/peterfazekas1999/LSTM-GAN-Prototype/blob/master/GAN_output.png" width ="300">
<img src="https://github.com/peterfazekas1999/LSTM-GAN-Prototype/blob/master/GAN_BTM.png" width ="300">

Below is the performance of the classifier trained on the Kaggle dataset, the output is 0 (No exo planet) and 1 (if there is an exop lanet)
Prediction 3 refers to the images above, without BATMAN, the classifier is not fooled. However after inserting artificial transits the classifier outputs 0.97, it is very confident that the lightcurve contains an exo planet.
<img src="https://github.com/peterfazekas1999/LSTM-GAN-Prototype/blob/master/prediction.png" width ="1000">


