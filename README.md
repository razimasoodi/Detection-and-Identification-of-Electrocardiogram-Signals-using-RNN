# Detection-and-Identification-of-Electrocardiogram-Signals-using-RNN
Electrocardiogram

An electrocardiogram records the electrical activity of the heart at each contraction. When an electric wave is generated in the heart, the inside of the heart cell quickly becomes positive relative to the outside. Stimulation by an electric wave nourishes the polarity of the cell.

![image](https://github.com/razimasoodi/Detection-and-Identification-of-Electrocardiogram-Signals-using-RNN/assets/170275013/3c2e5d5e-a0f5-47f8-b2be-7dd6068ec6ff)

The most important characteristic of ECG signal that cardiologists employ in diagnosis of heart disease is the QRS complex. This characteristic is more important than T and P waves and other signal properties because it is easier to distinguish and separate from the ECG signal than other
features. Also, QRS complex shows ventricular depolarization, which plays the most important role in the electrical activity of the heart. Therefore, the diagnosis and isolation of the compound is crucial in the classification and diagnosis of cardiac abnormalities. Also, by diagnosing and counting QRS, the intensity of the heartbeat and its possible inconsistencies can be observed and examined.

Dataset: QT database

In this assignment, you will predict the secondary sequences (annotations) from their primary sequences (samples). You are given a dataset of 146 files. Each sample file contains a primary sequence, ‘mV’ of signal, and its secondary is in annotation file ‘annotation’. Each primary sequence contains data streams of continuous values and its secondary sequence consist of 0 to 3 (1 for P wave, 2 for QRS complex, 3 for T wave and 0 for other points of signal). Thus, you have 73 input and output patterns of data streams. An example of input and output data streams is shown below.

![image](https://github.com/razimasoodi/Detection-and-Identification-of-Electrocardiogram-Signals-using-RNN/assets/170275013/ae886621-0e46-46e8-acd4-9b18b4233111)
![image](https://github.com/razimasoodi/Detection-and-Identification-of-Electrocardiogram-Signals-using-RNN/assets/170275013/bab5158c-a5fc-44dd-a037-7a0b74b5d385)

1. Use the first 80% of each data sequence files (primary and its secondary) for training and the rest 20% for test.
2. Use a sliding window of size 11 for each sequence to obtain the training samples.
3. Design, train and test an Elman neural network.
4. Design, train and test an NARX neural network.
5. Report the training and test accuracy of each method.
6. Use a sliding window of size 5 and 21 and repeat steps 3 to 5.


