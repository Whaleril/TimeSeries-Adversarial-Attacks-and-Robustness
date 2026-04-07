import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import os
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import tensorflow.keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def power_data():
    power_txt_path = '../Dataset/Power conumption dataset/household_power_consumption.txt'
    power_zip_path = '../Dataset/Power conumption dataset/household_power_consumption.zip'
    power_path = power_txt_path if os.path.isfile(power_txt_path) else power_zip_path

    df = pd.read_csv(power_path, sep=';',
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
                 low_memory=False, na_values=['nan','?'], index_col='dt')

    ## finding all columns that have nan:
    droping_list_all=[]
    for j in range(0,7):
        if not df.iloc[:, j].notnull().all():
            droping_list_all.append(j)
            #print(df.iloc[:,j].unique())

    # filling nan with mean in any columns
    for j in range(0,7):
            df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        dff = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(dff.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(dff.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    ## resampling of data over days
    df_resample = df.resample('h').mean()

    ## * Note: I scale all features in range of [0,1].

    ## If you would like to train based on the resampled data (over hour), then used below
    values = df_resample.values


    ## full data without resampling
    #values = df.values

    # integer encode direction
    # ensure all data is float
    #values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)

    # split into train and test sets
    values = reframed.values

    n_train_time = 365*72
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    ##test = values[n_train_time:n_test_time, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]


    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].

    return train_X, train_y, test_X, test_y , scaler


def window_slide(input_data, window_size):
    X = []
    y = []
    for i in range(len(input_data) - window_size - 1):
        t = []
        for j in range(0, window_size):
            t.append(input_data[[(i + j)], :])
        X.append(t)
        y.append(input_data[i + window_size, 1])
    return np.array(X), np.array(y)


def google_data():
    # import data
    stock_train = pd.read_csv("../Dataset/Google dataset/Google_train.csv",dtype={'Close': 'float64', 'Volume': 'int64','Open': 'float64','High': 'float64', 'Low': 'float64'})
    stock_test = pd.read_csv("../Dataset/Google dataset/Google_train.csv",dtype={'Close': 'float64', 'Volume': 'int64','Open': 'float64','High': 'float64', 'Low': 'float64'})

    stock_train.columns = ['date', 'close/last', 'volume', 'open', 'high', 'low']
    stock_test.columns = ['date', 'close/last', 'volume', 'open', 'high', 'low']

    #create a new column "average" 
    stock_train['average'] = (stock_train['high'] + stock_train['low'])/2
    stock_test['average'] = (stock_test['high'] + stock_test['low'])/2

    #pick the input features (average and volume)
    train_feature = stock_train.iloc[:,[2,6]].values
    train_data = train_feature

    test_feature= stock_test.iloc[:,[2,6]].values
    test_data = test_feature

    #data normalization
    sc= MinMaxScaler(feature_range=(0,1))
    train_data[:,0:2] = sc.fit_transform(train_feature[:,:])

    cs= MinMaxScaler(feature_range=(0,1))
    test_data[:,0:2] = cs.fit_transform(test_feature[:,:])

    scaler = [sc, cs]

    # data preparation
    lookback = 60

    train_X, train_y = window_slide(train_data, lookback)
    test_X, test_y = window_slide(test_data, lookback)

    train_X = train_X.reshape(train_X.shape[0], lookback, 2)
    test_X = test_X.reshape(test_X.shape[0],lookback, 2)

    return train_X, train_y, test_X, test_y , scaler


def traffic_data():
    preferred_path = "../Dataset/PEMS08-Traffic-Flow-Forecasting-main/traffic_dataset.csv"
    fallback_path = "../Dataset/PEMS08 traffic flow dataset/traffic_dataset.csv"
    traffic_path = preferred_path if os.path.isfile(preferred_path) else fallback_path
    df = pd.read_csv(traffic_path)

    values = df[["flow", "occupy", "speed"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    lookback = 12

    X, y = window_slide(scaled, lookback)

    train_end = int(0.7 * len(X))
    val_end = int(0.8 * len(X))

    train_X, train_y = X[:train_end], y[:train_end]
    val_X, val_y = X[train_end:val_end], y[train_end:val_end]
    test_X, test_y = X[val_end:], y[val_end:]

    train_X = train_X.reshape(train_X.shape[0], lookback, 3)
    val_X = val_X.reshape(val_X.shape[0], lookback, 3)
    test_X = test_X.reshape(test_X.shape[0], lookback, 3)

    return train_X, train_y, val_X, val_y, test_X, test_y, scaler

def compute_gradient(model_fn, loss_fn, x, y, targeted):
    """
    cleverhans : https://github.com/cleverhans-lab/cleverhans/blob/1115738a3f31368d73898c5bdd561a85a7c1c741/cleverhans/tf2/utils.py#L171

    Computes the gradient of the loss with respect to the input tensor.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param loss_fn: loss function that takes (labels, logits) as arguments and returns loss.
    :param x: input tensor
    :param y: Tensor with true labels. If targeted is true, then provide the target label.
    :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead try to move in the
                      direction of being more like y.
    :return: A tensor containing the gradient of the loss with respect to the input tensor.
    """

    with tf.GradientTape() as g:
        g.watch(x)
        # Compute loss
        loss = loss_fn(y, model_fn(x))
        if (
            targeted
        ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, x)
    return grad


def _to_numpy(value):
    if isinstance(value, tf.Tensor):
        return value.numpy()
    return np.array(value)


def compute_time_step_saliency(model_fn, loss_fn, x, y, targeted=False):
    ten_x = tf.convert_to_tensor(x)
    grad = compute_gradient(model_fn, loss_fn, ten_x, y, targeted)
    grad_np = _to_numpy(grad)
    saliency = np.sum(np.abs(grad_np), axis=2)
    return grad_np, saliency


def build_time_step_mask(saliency, top_ratio=0.1):
    saliency = np.array(saliency)
    batch_size, time_steps = saliency.shape
    top_k = max(1, int(np.ceil(time_steps * top_ratio)))
    mask = np.zeros_like(saliency, dtype=np.float32)

    for i in range(batch_size):
        key_indices = np.argsort(saliency[i])[-top_k:]
        mask[i, key_indices] = 1.0

    return mask[..., np.newaxis]


def perturbation_ratio(X, X_adv):
    delta = np.abs(np.array(X_adv) - np.array(X))
    changed_time_steps = np.any(delta > 1e-12, axis=2)
    return float(np.mean(changed_time_steps))


def fgsm(X, Y, model, loss_fn , epsilon, targeted= False):
    ten_X = tf.convert_to_tensor(X)
    grad = compute_gradient(model,loss_fn,ten_X,Y,targeted)
    dir = np.sign(_to_numpy(grad))
    return X + epsilon * dir, Y


def bim(X, Y, model, loss_fn, epsilon, alpha, I, targeted= False):
    Xp = np.copy(X)
    for t in range(I):
        ten_X = tf.convert_to_tensor(Xp)
        grad = compute_gradient(model,loss_fn,ten_X,Y,targeted)
        dir = np.sign(_to_numpy(grad))
        Xp = Xp + alpha * dir
        Xp = np.where(Xp > X+epsilon, X+epsilon, Xp)
        Xp = np.where(Xp < X-epsilon, X-epsilon, Xp)
    return Xp, Y


def ktsa_fgsm(X, Y, model, loss_fn, epsilon, top_ratio=0.1, targeted=False):
    grad_np, saliency = compute_time_step_saliency(model, loss_fn, X, Y, targeted)
    mask = build_time_step_mask(saliency, top_ratio=top_ratio)
    direction = np.sign(grad_np) * mask
    return X + epsilon * direction, Y, saliency, mask


def ktsa_bim(X, Y, model, loss_fn, epsilon, alpha, I, top_ratio=0.1, targeted=False):
    grad_np, saliency = compute_time_step_saliency(model, loss_fn, X, Y, targeted)
    mask = build_time_step_mask(saliency, top_ratio=top_ratio)
    Xp = np.copy(X)

    for t in range(I):
        ten_X = tf.convert_to_tensor(Xp)
        grad = compute_gradient(model, loss_fn, ten_X, Y, targeted)
        direction = np.sign(_to_numpy(grad)) * mask
        Xp = Xp + alpha * direction
        Xp = np.where(Xp > X + epsilon, X + epsilon, Xp)
        Xp = np.where(Xp < X - epsilon, X - epsilon, Xp)

    return Xp, Y, saliency, mask


def build_random_time_step_mask(batch_size, time_steps, top_ratio=0.1):
    """
    随机生成时间步掩码，数量严格对齐 KTSA 的 top_k。
    """
    top_k = max(1, int(np.ceil(time_steps * top_ratio)))
    mask = np.zeros((batch_size, time_steps), dtype=np.float32)

    for i in range(batch_size):
        # 在时间步中随机抽取 top_k 个索引，不重复
        random_indices = np.random.choice(time_steps, top_k, replace=False)
        mask[i, random_indices] = 1.0

    return mask[..., np.newaxis]


def random_fgsm(X, Y, model, loss_fn, epsilon, top_ratio=0.1, targeted=False):
    """
    基于随机选择时间步的 FGSM 攻击
    """
    batch_size, time_steps, _ = X.shape
    mask = build_random_time_step_mask(batch_size, time_steps, top_ratio=top_ratio)

    ten_X = tf.convert_to_tensor(X)
    grad = compute_gradient(model, loss_fn, ten_X, Y, targeted)
    # 取梯度方向并应用随机掩码
    direction = np.sign(_to_numpy(grad)) * mask
    return X + epsilon * direction, Y, mask


def random_bim(X, Y, model, loss_fn, epsilon, alpha, I, top_ratio=0.1, targeted=False):
    """
    基于随机选择时间步的 BIM 迭代攻击
    """
    batch_size, time_steps, _ = X.shape
    mask = build_random_time_step_mask(batch_size, time_steps, top_ratio=top_ratio)
    Xp = np.copy(X)

    for t in range(I):
        ten_X = tf.convert_to_tensor(Xp)
        grad = compute_gradient(model, loss_fn, ten_X, Y, targeted)
        direction = np.sign(_to_numpy(grad)) * mask
        Xp = Xp + alpha * direction
        Xp = np.where(Xp > X + epsilon, X + epsilon, Xp)
        Xp = np.where(Xp < X - epsilon, X - epsilon, Xp)

    return Xp, Y, mask