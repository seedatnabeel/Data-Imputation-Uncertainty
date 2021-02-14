"""
Written by Jinsung Yoon
Date: Jan 29th 2019
Generative Adversarial Imputation Networks (GAIN) Implementation on Spam Dataset
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN.pdf
Appendix Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN_Supp.pdf
Contact: jsyoon0823@g.ucla.edu

Modified by: Nabeel Seedat
Date: Feb 2021
"""

#%% Packages
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import initpath_alg

initpath_alg.init_sys_path()
import utilmlab
import data_loader_mlab
import matplotlib.pyplot as plt


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default="imputed.csv", help="output (csv) file")
    parser.add_argument("--it", default=50, type=int, help="iterations")
    parser.add_argument(
        "--dataset",
        help="load one of the available/buildin datasets"
        " [spam, spambase, letter, ...] use show to see a list",
    )
    parser.add_argument(
        "-i",
        help="load data as a csv file, requires the name of the label"
        " (reponsevar) to be specified as well (if applicable), this column"
        "will not be processed",
    )
    parser.add_argument(
        "--target",
        help="specifies the column with the response var "
        "if applicable when loading a csv file, this column will"
        " not be processed",
    )
    parser.add_argument("--testall", type=int, default=1)
    parser.add_argument("--ref")
    parser.add_argument("--bs", default=128, type=int, help="batch size")
    parser.add_argument("--pmiss", default=0.2, type=float, help="missing rate")
    parser.add_argument("--phint", default=0.5, type=float, help="hint rate")
    parser.add_argument("--alpha", default=10, type=float, help="")
    parser.add_argument("--autocategorical", default=1, type=int, help="")
    parser.add_argument("--verbose", default=0, type=int, help="")
    parser.add_argument("--trainratio", default=0.8, type=float, help="")
    parser.add_argument("--samples", default=50, type=int, help="")

    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()

    dataset = args.dataset

    import os

    folder_path = f"./data/imputed/{dataset}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fn_icsv = args.i
    fn_ref_csv = args.ref
    fn_ocsv = f"./data/imputed/{dataset}/" + args.o
    odir = os.path.dirname(fn_ocsv)
    odir = odir if len(odir) else "."
    logger = utilmlab.init_logger(odir)

    mb_size = args.bs
    p_hint = args.phint
    alpha = args.alpha
    train_rate = args.trainratio
    niter = args.it
    test_all = args.testall
    label = args.target
    is_auto_categorical = args.autocategorical
    is_cat_one_hot = args.autocategorical == 2
    is_verbose = args.verbose

    samples = args.samples

    folder_path = f"./data/interim/{dataset}/"

    df = pd.read_csv(f"./data/raw/{dataset}/" + fn_icsv)
    features = list(df.columns)

    if label is not None:
        if label not in features:
            assert label in features
        features.remove(label)

    if fn_ref_csv is not None:
        df_ref = pd.read_csv(fn_ref_csv)

    if is_auto_categorical:
        df_tmp, prop_df_one_hot = utilmlab.df_cat_to_one_hot(
            df[features], is_verbose=is_verbose, is_cat_one_hot=is_cat_one_hot
        )

    # Load both elements of the tuple (data, missing)
    Data = np.array(pd.read_csv(folder_path + "Datadf.csv"))
    Missing = np.array(pd.read_csv(folder_path + "Missingdf.csv"))

    # Parameters
    No = len(Data)
    Dim = len(Data[0, :])

    # Hidden state dimensions
    H_Dim1 = Dim
    H_Dim2 = Dim

    Train_No = int(No * train_rate)
    Test_No = No - Train_No

    # Read the created datasets to train and test model on
    trainX = np.array(pd.read_csv(folder_path + "trainX.csv"))
    trainM = np.array(pd.read_csv(folder_path + "trainM.csv"))
    testX = np.array(pd.read_csv(folder_path + "testX.csv"))
    testM = np.array(pd.read_csv(folder_path + "testM.csv"))

    import pickle

    # load the scaler back
    PIK = folder_path + "scaler.p"
    with open(PIK, "rb") as f:
        scaler = pickle.load(f)

    # 1. Xavier Initialization Definition
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    # Hint Vector Generation
    def sample_M(m, n, p):
        if p == 0:
            B = np.ones((m, n))
            for i in range(m):
                B[np.random.randint(0, n)] = 0
        else:
            A = np.random.uniform(0.0, 1.0, size=[m, n])
            B = A > p
        C = 1.0 * B
        return C

    """
    GAIN Consists of 3 Components
    - Generator
    - Discriminator
    - Hint Mechanism
    """

    #%% GAIN Architecture

    #%% 1. Input Placeholders
    # 1.1. Data Vector
    X = tf.placeholder(tf.float32, shape=[None, Dim])
    New_X = tf.placeholder(tf.float32, shape=[None, Dim])
    # 1.2. Mask Vector
    M = tf.placeholder(tf.float32, shape=[None, Dim])
    # 1.4. X with missing values
    X_R = tf.placeholder(tf.float32, shape=[None, Dim])
    X_F = tf.placeholder(tf.float32, shape=[None, Dim])
    M_R = tf.placeholder(tf.float32, shape=[None, Dim])

    #%% 2. Discriminator

    use_default_GAIN_model = True
    if use_default_GAIN_model:
        D_W1 = tf.Variable(xavier_init([2 * Dim, H_Dim1]))  # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

        D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
        D_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))
        D_W3 = tf.Variable(xavier_init([H_Dim2, 1]))
        D_b3 = tf.Variable(tf.zeros(shape=[1]))  # Output is multi-variate

        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

        G_W1 = tf.Variable(
            xavier_init([Dim * 2, H_Dim1])
        )  # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

        G_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
        G_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

        G_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
        G_b3 = tf.Variable(tf.zeros(shape=[Dim]))

        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        #%% GAIN Function

        #%% 1. Generator
        def generator(new_x, m):
            inputs = tf.concat(axis=1, values=[new_x, m])  # Mask + Data Concatenate
            G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
            G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
            G_prob = tf.nn.sigmoid(
                tf.matmul(G_h2, G_W3) + G_b3
            )  # [0,1] normalized Output

            return G_prob

        #%% 2. Discriminator
        def discriminator(new_x, h):
            inputs = tf.concat(axis=1, values=[new_x, h])  # Hint + Data Concatenate
            D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
            D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
            D_logit = tf.matmul(D_h2, D_W3) + D_b3
            D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output

            return D_prob

    else:  # implement different network here
        D_W1 = tf.Variable(xavier_init([2 * Dim, H_Dim1]))  # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

        D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
        D_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))
        D_W3 = tf.Variable(xavier_init([H_Dim2, 1]))
        D_b3 = tf.Variable(tf.zeros(shape=[1]))  # Output is multi-variate
        D_W4 = tf.Variable(xavier_init([H_Dim2, H_Dim2]))
        D_b4 = tf.Variable(tf.zeros(shape=[H_Dim2]))  # Output is multi-variate

        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3, D_W3, D_b3, D_b4, D_W4]

        G_W1 = tf.Variable(xavier_init([Dim * 2, H_Dim1]))
        G_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

        G_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
        G_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))
        G_W4 = tf.Variable(xavier_init([H_Dim2, H_Dim2]))
        G_b4 = tf.Variable(tf.zeros(shape=[H_Dim2]))

        G_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
        G_b3 = tf.Variable(tf.zeros(shape=[Dim]))

        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3, G_W4, G_b4]

        #%% GAIN Function

        #%% 1. Generator
        def generator(new_x, m):
            inputs = tf.concat(axis=1, values=[new_x, m])  # Mask + Data Concatenate
            G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
            G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
            G_h3 = tf.nn.relu(tf.matmul(G_h2, D_W4) + D_b4)
            G_prob = tf.nn.sigmoid(
                tf.matmul(G_h3, G_W3) + G_b3
            )  # [0,1] normalized Output

            return G_prob

        #%% 2. Discriminator
        def discriminator(new_x, h):
            inputs = tf.concat(axis=1, values=[new_x, h])  # Hint + Data Concatenate
            D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
            D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
            D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W4) + D_b4)
            D_logit = tf.matmul(D_h3, D_W3) + D_b3
            D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output

            return D_prob

    #%% 3. Other functions
    # Random sample generator for Z
    def sample_Z(m, n):
        return np.random.uniform(0.0, 0.01, size=[m, n])

    # Mini-batch generation
    def sample_idx(m, n):
        A = np.random.permutation(m)
        idx = A[:n]
        return idx

    #%% Structure
    # Generator
    G_sample = generator(New_X, M)

    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1 - M)
    X_F = Hat_New_X * M_R

    # Discriminator
    D_fake = discriminator(X_F, M_R)
    D_real = discriminator(X_R, M_R)

    #%% Loss
    D_loss1 = -tf.reduce_mean(tf.log(D_real + 1e-8) + tf.log(1.0 - D_fake + 1e-8))
    G_loss1 = -tf.reduce_mean(tf.log(D_fake + 1e-8))

    MSE_train_loss = tf.reduce_mean((M * New_X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss1
    G_loss = G_loss1 + alpha * MSE_train_loss

    #%% MSE Performance metric
    MSE_test_loss = tf.reduce_mean(
        ((1 - M) * X - (1 - M) * G_sample) ** 2
    ) / tf.reduce_mean(1 - M)

    #%% Solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #%% Iterations

    #%% Start Iterations

    pbar = tqdm(range(niter))
    for it in pbar:

        #%% Inputs
        mb_idx = sample_idx(Train_No, mb_size)  # random idxs of mbsize
        X_mb = trainX[mb_idx, :]

        Z_mb = sample_Z(mb_size, Dim)  # random noise between 0 and 0.01
        M_mb = trainM[mb_idx, :]  # mask mbsize
        mb_idx2 = sample_idx(Train_No, mb_size)  # random idxs of mbsize
        X_mbR = trainX[mb_idx2, :]
        M_mbR = trainM[mb_idx2, :]

        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

        _, D_loss_curr = sess.run(
            [D_solver, D_loss1],
            feed_dict={M: M_mb, X: X_mb, New_X: New_X_mb, X_R: X_mbR, M_R: M_mbR},
        )
        _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run(
            [G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
            feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, X_R: X_mbR, M_R: M_mbR},
        )
        #%% Intermediate Losses
        if it % 100 == 0:
            s = "{:6d}) loss train {:0.3f} test {:0.3f}".format(
                it, np.sqrt(MSE_train_loss_curr), np.sqrt(MSE_test_loss_curr)
            )
            pbar.clear()
            logger.info("{}".format(s))
            pbar.set_description(s)

    # Monte Carlo sample (i.e. we need to sample latent Z T times)
    gain_samples = []
    for i in tqdm(range(samples)):
        if not test_all:
            Z_mb = sample_Z(Test_No, Dim)
            M_mb = testM
            X_mb = testX
        else:
            Z_mb = sample_Z(No, Dim)
            M_mb = Missing
            X_mb = Data
            if fn_ref_csv:
                testX = df_ref[features].values
            else:
                testX = Data
            testM = Missing

            logger.info(
                "testall: {} {} {} {}".format(
                    Z_mb.shape, M_mb.shape, X_mb.shape, New_X_mb.shape
                )
            )

        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

        MSE_final, Sample = sess.run(
            [MSE_test_loss, G_sample], feed_dict={X: testX, M: testM, New_X: New_X_mb}
        )

        testX_imputed = np.where(testM < 1, Sample, testX)

        gain_samples.append(testX_imputed)

        testX_imputed = scaler.inverse_transform(testX_imputed)

    if is_auto_categorical:
        testX_imputed = utilmlab.df_one_hot_to_cat(
            pd.DataFrame(testX_imputed, columns=prop_df_one_hot["dfcol_one_hot"]),
            prop_df_one_hot,
        )

    df_imputed = pd.DataFrame(testX_imputed, columns=features)
    if label is not None:
        df_imputed[[label]] = df[[label]]

    df_imputed.to_csv(fn_ocsv, index=False)

    import pickle

    # serialize the GAIN results for later analysis
    PIK = f"./data/imputed/{dataset}/" + "gain_samples.p"
    with open(PIK, "wb") as f:
        pickle.dump(gain_samples, f)
