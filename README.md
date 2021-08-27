# FDN-Learning
Our model utilized the muti-cities historical air pollutant and meteorological data  to predict future pollutant concentration

the training data validation data and test data are priviate;

and the data sample has been upload in my perssonel page：https://github.com/zouguojian/data

the whole codes have been upload in this page web, you can download to check it.

if you have any question, you can sent e-mails to me, e-m: 873374677@qq.com

Abstract— The problem of increasing air pollution poses a challenge to smart city development, as spatial air pollution correlation exists among adjacent cities. However, it is difficult to predict the degree of air pollution of a location by exploiting massive air pollution datasets incorporating data on spatially related locations. Construction of a spatial correlation prediction model for air pollution is therefore required for air pollution big-data mining. In this paper, we propose an air pollution-concentration spatial correlation prediction model based on a fusion deep neural network called FDN-learning. Three models are combined: a stacked anti-autoencoder network, Gaussian function model, and long short-term memory network (LSTM). The FDN-learning model is composed of three layers for feature expansion, intermediate processing, and data prediction. In the first layer, we employ a stacked anti-autoencoder model to learn the source-data spatial features through a feature expansion hidden layer; this can enrich the feature vector and mine more information for further prediction. In the second layer, the Gaussian function evaluates effective weights for the outputs of the stacked anti-autoencoder models in the preceding layer; the spatial correction effects are therefore incorporated in this layer. Finally, the LSTM model in the data prediction layer learns the air pollution-concentration temporal features. A fine-tuning method based on stochastic gradient descent is applied to the FDN-learning model for improved performance. Empirical results are used to verify the feasibility and effectiveness of our proposed model based on a real-world air pollution dataset.
Keywords—Fusion deep neural network, Gaussian function, LSTM, PM2.5-concentration Prediction, Stacked anti-autoencoder
