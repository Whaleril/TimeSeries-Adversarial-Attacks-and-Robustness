import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import power_data, fgsm, bim, ktsa_fgsm, ktsa_bim, perturbation_ratio, rmse
from model import setup_lstm_model

## for Deep-learing:
import tensorflow.keras.backend as K
# from tensorflow.keras.layers import Dense, GRU

## Data can be downloaded from: http://archive.ics.uci.edu/ml/machine-learning-databases/00235/
## Just open the zip file and grab the file 'household_power_consumption.txt' put it in the directory
## that you would like to run the code.

model_path = '../Trained models/Power_regression_LSTM.h5'
epsilon = 0.2
alpha = 0.001
iterations = 200
ktsa_top_ratio = 0.1


train_X, train_y, test_X, test_y , scaler = power_data()


def summarize_metrics(name, actual, predicted):
	rmse_value = np.sqrt(mean_squared_error(actual, predicted))
	mae_value = mean_absolute_error(actual, predicted)
	print('%s RMSE: %.3f' % (name, rmse_value))
	print('%s MAE: %.3f' % (name, mae_value))
	return {"rmse": rmse_value, "mae": mae_value}


if os.path.isfile(model_path):
	model = setup_lstm_model(train_X, 100)
	model.load_weights(model_path)
	test_X_seq = np.copy(test_X)

	# make adversarial example
	adv_X, _ = fgsm(X=test_X_seq, Y=test_y, model=model, loss_fn=rmse, epsilon=epsilon)
	bim_X, _ = bim(X=test_X_seq, Y=test_y, model=model, loss_fn=rmse, epsilon=epsilon, alpha=alpha, I=iterations)
	ktsa_adv_X, _, ktsa_saliency, ktsa_mask = ktsa_fgsm(
		X=test_X_seq,
		Y=test_y,
		model=model,
		loss_fn=rmse,
		epsilon=epsilon,
		top_ratio=ktsa_top_ratio,
	)
	ktsa_bim_X, _, _, _ = ktsa_bim(
		X=test_X_seq,
		Y=test_y,
		model=model,
		loss_fn=rmse,
		epsilon=epsilon,
		alpha=alpha,
		I=iterations,
		top_ratio=ktsa_top_ratio,
	)

	# make a adv prediction
	adv_yhat = model.predict(adv_X)
	bim_yhat = model.predict(bim_X)
	ktsa_adv_yhat = model.predict(ktsa_adv_X)
	ktsa_bim_yhat = model.predict(ktsa_bim_X)
	adv_X = adv_X.reshape((adv_X.shape[0], 7))
	bim_X = bim_X.reshape((bim_X.shape[0], 7))
	ktsa_adv_X = ktsa_adv_X.reshape((ktsa_adv_X.shape[0], 7))
	ktsa_bim_X = ktsa_bim_X.reshape((ktsa_bim_X.shape[0], 7))
	# make a prediction
	yhat = model.predict(test_X_seq)
	test_X = test_X_seq.reshape((test_X_seq.shape[0], 7))

	# invert scaling for adv forecast
	inv_adv_yhat = np.concatenate((adv_yhat, test_X[:, -6:]), axis=1)
	inv_adv_yhat = scaler.inverse_transform(inv_adv_yhat)
	inv_adv_yhat = inv_adv_yhat[:,0]
	# invert scaling for bim forecast
	inv_bim_yhat = np.concatenate((bim_yhat, test_X[:, -6:]), axis=1)
	inv_bim_yhat = scaler.inverse_transform(inv_bim_yhat)
	inv_bim_yhat = inv_bim_yhat[:,0]
	# invert scaling for ktsa forecast
	inv_ktsa_adv_yhat = np.concatenate((ktsa_adv_yhat, test_X[:, -6:]), axis=1)
	inv_ktsa_adv_yhat = scaler.inverse_transform(inv_ktsa_adv_yhat)
	inv_ktsa_adv_yhat = inv_ktsa_adv_yhat[:,0]
	# invert scaling for ktsa bim forecast
	inv_ktsa_bim_yhat = np.concatenate((ktsa_bim_yhat, test_X[:, -6:]), axis=1)
	inv_ktsa_bim_yhat = scaler.inverse_transform(inv_ktsa_bim_yhat)
	inv_ktsa_bim_yhat = inv_ktsa_bim_yhat[:,0]
	# invert scaling for forecast
	inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]

	# invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]

	clean_metrics = summarize_metrics('Clean', inv_y, inv_yhat)
	fgsm_metrics = summarize_metrics('FGSM', inv_y, inv_adv_yhat)
	bim_metrics = summarize_metrics('BIM', inv_y, inv_bim_yhat)
	ktsa_fgsm_metrics = summarize_metrics('KTSA-FGSM', inv_y, inv_ktsa_adv_yhat)
	ktsa_bim_metrics = summarize_metrics('KTSA-BIM', inv_y, inv_ktsa_bim_yhat)

	print('FGSM perturbation ratio: %.3f' % perturbation_ratio(test_X_seq, adv_X.reshape((-1, 1, 7))))
	print('BIM perturbation ratio: %.3f' % perturbation_ratio(test_X_seq, bim_X.reshape((-1, 1, 7))))
	print('KTSA-FGSM perturbation ratio: %.3f' % perturbation_ratio(test_X_seq, ktsa_adv_X.reshape((-1, 1, 7))))
	print('KTSA-BIM perturbation ratio: %.3f' % perturbation_ratio(test_X_seq, ktsa_bim_X.reshape((-1, 1, 7))))
	print('KTSA top ratio: %.3f' % ktsa_top_ratio)
	print('Average selected key-step ratio from mask: %.3f' % float(np.mean(ktsa_mask)))
	print('Average KTSA saliency score: %.6f' % float(np.mean(ktsa_saliency)))

	print('FGSM RMSE increase over Clean: %.3f' % (fgsm_metrics["rmse"] - clean_metrics["rmse"]))
	print('BIM RMSE increase over Clean: %.3f' % (bim_metrics["rmse"] - clean_metrics["rmse"]))
	print('KTSA-FGSM RMSE increase over Clean: %.3f' % (ktsa_fgsm_metrics["rmse"] - clean_metrics["rmse"]))
	print('KTSA-BIM RMSE increase over Clean: %.3f' % (ktsa_bim_metrics["rmse"] - clean_metrics["rmse"]))
	print('KTSA-FGSM RMSE delta vs FGSM: %.3f' % (ktsa_fgsm_metrics["rmse"] - fgsm_metrics["rmse"]))
	print('KTSA-BIM RMSE delta vs BIM: %.3f' % (ktsa_bim_metrics["rmse"] - bim_metrics["rmse"]))
	print('KTSA-FGSM MAE delta vs FGSM: %.3f' % (ktsa_fgsm_metrics["mae"] - fgsm_metrics["mae"]))
	print('KTSA-BIM MAE delta vs BIM: %.3f' % (ktsa_bim_metrics["mae"] - bim_metrics["mae"]))

	## time steps, every step is one hour (you can easily convert the time step to the actual time index)
	## for a demonstration purpose, I only compare the predictions in 200 hours.

	fig_verify = plt.figure(figsize=(100, 50))
	aa=[x for x in range(200)]
	plt.plot(aa, inv_y[:200], marker='.', label="actual")
	plt.plot(aa, inv_yhat[:200], 'r', label="prediction")
	plt.plot(aa, inv_adv_yhat[:200], 'g', label="fgsm prediction")
	plt.plot(aa, inv_bim_yhat[:200], 'b', label="bim prediction")
	plt.plot(aa, inv_ktsa_adv_yhat[:200], 'm', label="ktsa-fgsm prediction")
	plt.plot(aa, inv_ktsa_bim_yhat[:200], 'c', label="ktsa-bim prediction")
	plt.ylabel('Global_active_power', size=15)
	plt.xlabel('Time step', size=15)
	plt.legend(fontsize=15)
	fig_verify.savefig("../Images/Power_LSTM_test_plot.png")
	plt.close(fig_verify)
