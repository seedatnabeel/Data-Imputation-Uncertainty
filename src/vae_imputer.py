import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import initpath_alg

initpath_alg.init_sys_path()


from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.layers import Input, Dense, Dropout


class MC_VAE_Imputer:
    def __init__(
        self,
        weight_update_coeff=0.4,
        optimizer="adam",
        dropout_probability=0.2,
        activation_func="relu",
        output_activation="sigmoid",
        weight_init="glorot_normal",
        hidden_size=1000,
        n_dims=100,
        mcd=True,
        mc_samples=100,
    ):

        self.n_dims = n_dims
        self.weight_update_coeff = weight_update_coeff
        self.keep_coeff = 1 - self.weight_update_coeff
        self.optimizer = optimizer
        self.dropout_probability = dropout_probability
        self.activation_func = activation_func
        self.output_activation = output_activation
        self.weight_init = weight_init
        self.hidden_size = hidden_size
        self.mcd = mcd

        if self.mcd == False:
            self.mc_samples = 1
        else:
            self.mc_samples = mc_samples

        self._build_and_compile_vae()

    def _encoder(self, z_dimension, mcd):
        """
        Define the Encoder architecture.
        Args:
            z_dimension (int): size of latent dim
            mcd (bool): whether to apply mcd
        """
        inputs = Input(shape=(2 * self.n_dims,), name="encoder_input")
        x = inputs
        x = Dense(
            self.hidden_size,
            activation=self.activation_func,
            kernel_initializer=self.weight_init,
        )(x)
        x = Dropout(self.dropout_probability)(x, training=mcd)
        x = Dense(
            self.hidden_size // 2,
            activation=self.activation_func,
            kernel_initializer=self.weight_init,
        )(x)
        x = Dropout(self.dropout_probability)(x, training=mcd)

        x = Dense(
            self.hidden_size // 3,
            activation=self.activation_func,
            kernel_initializer=self.weight_init,
        )(x)
        x = Dropout(self.dropout_probability)(x, training=mcd)

        x = Dense(
            self.hidden_size // 4,
            activation=self.activation_func,
            kernel_initializer=self.weight_init,
        )(x)
        x = Dropout(self.dropout_probability)(x, training=mcd)

        # Latent mean
        z_mean = Dense(z_dimension, name="z_mean")(x)
        
        # latent variance
        z_var = Dense(z_dimension, name="z_var")(x)
        self.encoder = Model(inputs, [z_mean, z_var], name="encoder")

        return self.encoder, inputs, z_mean, z_var

    def _decoder(self, inputs, z_dimension, mcd):
        """[summary]
        Define the Decoder architecture.
        Args:
            inputs: defines the keras input data
            z_dimension (int): size of latent dim
            mcd (bool): whether to apply mcd

        """
        
        latent_inputs = Input(shape=(z_dimension,), name="z_sampling")
        x = latent_inputs
        x = Dense(
            self.hidden_size // 4,
            activation=self.activation_func,
            kernel_initializer=self.weight_init,
        )(x)
        x = Dropout(self.dropout_probability)(x, training=mcd)

        x = Dense(
            self.hidden_size // 3,
            activation=self.activation_func,
            kernel_initializer=self.weight_init,
        )(x)
        x = Dropout(self.dropout_probability)(x, training=mcd)

        x = Dense(
            self.hidden_size // 2,
            activation=self.activation_func,
            kernel_initializer=self.weight_init,
        )(x)
        x = Dropout(self.dropout_probability)(x, training=mcd)

        x = Dense(
            self.hidden_size,
            activation=self.activation_func,
            kernel_initializer=self.weight_init,
        )(x)
        x = Dropout(self.dropout_probability)(x, training=mcd)
        outputs = Dense(
            self.n_dims,
            activation=self.output_activation,
            kernel_initializer=self.weight_init,
        )(x)
        
        self.decoder = Model(latent_inputs, outputs, name="decoder")
        
        outputs = self.decoder(self.encoder(inputs)[0])
        
        return self.decoder, outputs

    def _build_and_compile_vae(self):
        """
        Builds and compiles Keras model
        """
        z_dimension = int(self.n_dims//2)+1

        self.encoder, inputs, z_mean, z_var = self._encoder(z_dimension, self.mcd)

        self.decoder, outputs = self._decoder(inputs, z_dimension, self.mcd)

        self.model = Model(inputs, outputs, name="VAE IMPUTER")

        loss = self.vae_loss(self.n_dims, z_mean, z_var)

        self.model.compile(optimizer=self.optimizer, loss=loss)

    def vae_loss(self, n_features, z_mean, z_var):
        """
        Custom VAE loss
        """
        def reconstruction_loss(input_and_mask, preds):
            # Split the data and the missing mask tuple
            data = input_and_mask[:, :n_features]
            missing_mask = input_and_mask[:, n_features:]
            
            true_X = data * (1 - missing_mask)
            pred_X = preds * (1 - missing_mask)

            # compute the reconstruction loss
            reconstruction_loss = binary_crossentropy(true_X, pred_X)
            reconstruction_loss *= n_features
            
            # compute the KL divergence
            kl_div = 1 + z_var - K.square(z_mean) - K.exp(z_var)
            kl_div = K.sum(kl_div, axis=-1)
            
            return K.mean(reconstruction_loss -1/2 * kl_div)

        return reconstruction_loss

    def predict(self, x_test_with_mask):
        # Pass data through the encoder to get latent Z
        latent_z = self.encoder.predict(x_test_with_mask)[0]

        # Apply MC-Dropout
        decoder_prediction = K.function(
            [self.decoder.layers[0].input, K.learning_phase()],
            [self.decoder.layers[-1].output],
        )

        pred_shape_dim1, pred_shape_dim2 = x_test_with_mask.shape[0], x_test_with_mask.shape[1] // 2        
    
        # MC sampling
        outputs = []
        for _ in range(self.mc_samples):
            prediction = np.array(decoder_prediction([latent_z, 1])).reshape(
                (pred_shape_dim1, pred_shape_dim2)
            )
            outputs.append(prediction)

        outputs = np.array(outputs)
        return np.mean(outputs, axis=0), outputs

    def parameter_update(self, X, X_mask):
        """
        Applies the parameter updates and does the sampling with
        
        Args: 
        X (np.array): input data
        X_mask (np.array): missing mask
        
        Returns: 
        X (np.array): updated input data
        X_MC_preds (np.array): MC samples
        """
        data_with_mask = np.hstack([X, X_mask])
        X_pred, X_MC_preds = self.predict(data_with_mask)
        X[X_mask] *= self.keep_coeff
        X[X_mask] += self.weight_update_coeff * X_pred[X_mask]
        return X, X_MC_preds

    def fit(
        self,
        x_train,
        x_test,
        missing_mask,
        x_test_missing_mask,
        batch_size,
        epochs=100,
    ):
        """Fits the VAE

        Args:
            x_train (np.array): train data
            x_test (np.array): test data
            missing_mask (np.array): train missing mask
            x_test_missing_mask (np.array): test missing mask
            batch_size (int): training batch size
            epochs (int, optional): Num epochs. Defaults to 100.

        Returns:
            x_train (np.array): imputed train data
            x_test (np.array): imputed test data
            X_test_MC_preds (np.array): MC samples test
        """
        for epoch in range(epochs):
            print(f"Epoch: [{epoch}/{epochs}]")

            input_with_mask = np.hstack([x_train, missing_mask])
            self.model.fit(x=input_with_mask, y=input_with_mask, batch_size=batch_size)

            # Update x_train
            x_train, _ = self.parameter_update(x_train, missing_mask)

            # Update x_test
            x_test, X_test_MC_preds = self.parameter_update(x_test, x_test_missing_mask)

        return x_train, x_test, X_test_MC_preds
