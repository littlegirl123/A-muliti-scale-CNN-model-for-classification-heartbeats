# A muliti-scale CNN model for classification heartbeats

 We bulid a 15 layer multi-scale cnn model to divide the ECG data into N, S, V and F.

Materials
MIT Arrhythmia database: we choose 300 sampling points, 100 sampling points before R-peak and 200 sampling points after it.
RR intervals: pre-RR-interval、post-RR-interval and average RR interval

Model
The proposed CNN model consists of four parts: input block, multi-scale block, feature extraction block and classification. 
The input block concatenates the original ECG signal x and frequency domain features ωx which are obtained through wavelet transform as the input of multi-scale block.
In the multi-scale block, we employ convolution filters with different sizes to extract scale-relevant information. And then we concatenate the different scales morphological information as the input of feature extraction block.
As for the classification, we concatenate the RR intervals and morphological features extracted by the multi-scale model as the final ECG features for classification.


