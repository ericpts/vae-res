import tensorflow.keras as keras
import tensorflow as tf
import time
from config import *

from vae import VAE


class SuperVAE(tf.keras.Model):

    def __init__(self, latent_dim: int, name: str) -> None:
        super(SuperVAE, self).__init__(name=name)

        self.latent_dim = latent_dim

        self.nvaes = config.nvaes

        inputs = keras.Input(
            shape=(28 * config.expand_per_height, 28 * config.expand_per_width,
                   1))

        self.vaes = [
            VAE(latent_dim, 'VAE-{}'.format(i)) for i in range(self.nvaes)
        ]

        self.vae_is_learning = {i: True for i in range(self.nvaes)}

        vae_images = []
        vae_confidences = []
        for vae in self.vaes:
            (image, confidence) = vae(inputs)
            vae_images.append(image)
            vae_confidences.append(confidence)

        vae_confidences = tf.convert_to_tensor(vae_confidences)
        vae_images = tf.convert_to_tensor(vae_images)

        softmax_confidences = tf.keras.layers.Softmax(axis=0)(vae_confidences)

        self.model = keras.models.Model(
            inputs=inputs, outputs=[softmax_confidences, vae_images])
        self.set_lr(lr = 0.001)


    def set_lr(self, lr: float):
        self.lr = lr

        self.fast_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.lr,
                )

        # This optimizer has a very small learning rate, so that frozen VAE's
        # can still adapt, but at a much slower pace than the ones which are
        # supposed to be actively learning.
        self.slow_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.lr * 1e-9,
                )



    def unfreeze_vae(self, i: int):
        assert i in self.vae_is_learning
        self.vae_is_learning[i] = True

    def freeze_vae(self, i: int):
        assert i in self.vae_is_learning
        self.vae_is_learning[i] = False


    @tf.function
    def entropy_loss(self, softmax_confidences):
        entropy = - tf.math.xlogy(softmax_confidences, softmax_confidences)
        entropy = tf.math.reduce_sum(entropy, axis=0)
        entropy /= tf.math.log(float(self.nvaes))
        entropy = -tf.math.log(entropy)
        entropy = tf.math.reduce_sum(entropy, axis=[1, 2])
        return entropy


    @tf.function
    def compute_loss(self, x):
        (softmax_confidences, vae_images) = self.model(x)

        loss_object = tf.keras.losses.MeanSquaredError()
        recall_loss = 0.0
        for i in range(self.nvaes):
            recall_loss += loss_object(
                x, vae_images[i], sample_weight=softmax_confidences[i])
        recall_loss /= self.nvaes
        recall_loss *= 28 * 28 * config.expand_per_width * config.expand_per_height

        kl_loss = 0.0
        for nvae in self.vaes:
            kl_loss += VAE.compute_kl_loss(nvae.last_mean, nvae.last_logvar)
        kl_loss /= self.nvaes

        ent_loss = self.entropy_loss(softmax_confidences)

        vae_loss = tf.math.reduce_mean(
            recall_loss + config.beta * kl_loss + config.gamma * ent_loss)
        return vae_loss


    def get_trainable_variables(self):
        return self.model.trainable_variables


    def apply_gradients(self, grads_per_vae):
        for i in range(self.nvaes):
            optimizer = None
            if self.vae_is_learning[i]:
                optimizer = self.fast_optimizer
            else:
                optimizer = self.slow_optimizer
            optimizer.apply_gradients(
                    zip(
                        grads_per_vae[i],
                        self.vaes[i].get_trainable_variables()
                        )
                    )


    @tf.function
    def compute_gradients(self, X):
        variables = self.get_trainable_variables()
        with tf.GradientTape() as tape:
            loss = self.compute_loss(X)
        return tape.gradient(loss, variables), loss


    def fit(self, D_train: tf.data.Dataset):
        def partition_gradients(grads):
            """ Returns the gradients for each VAE.
            """
            ret = []
            at = 0
            for i in range(self.nvaes):
                nvars = len(self.vaes[i].get_trainable_variables())
                cur_grads = grads[at : at + nvars]
                ret.append(cur_grads)
                at += nvars

            assert len(grads) == len(self.get_trainable_variables())
            assert at == len(grads)
            assert len(ret) == self.nvaes
            return ret

        train_loss = 0
        train_size = 0

        for (X, y) in D_train.batch(
                config.batch_size, drop_remainder=True).prefetch(
                    4 * config.batch_size
                ):
            gradients, loss = self.compute_gradients(X)

            grads_per_vae = partition_gradients(gradients)

            self.apply_gradients(grads_per_vae)

            train_loss += loss * X.shape[0]
            train_size += X.shape[0]

        return train_loss / train_size


    def evaluate_on_dataset(self, D_test):
        test_loss = 0
        test_size = 0
        for (X, y) in D_test.batch(
                config.batch_size * 8, drop_remainder=True):
            test_loss += self.compute_loss(X) * X.shape[0]
            test_size += X.shape[0]

        return test_loss / test_size

    def run_on_input(self, X):
        (softmax_confidences, vae_images) = self.model(X)
        return (softmax_confidences, vae_images)


