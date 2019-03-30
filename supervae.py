import tensorflow.keras as keras
import tensorflow as tf
import time
from config import *
import numpy as np

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

        self.vae_is_learning = np.array([True for i in range(self.nvaes)])

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
        assert 0 <= i and i < self.nvaes
        self.vae_is_learning[i] = True

    def freeze_vae(self, i: int):
        assert 0 <= i and i < self.nvaes
        self.vae_is_learning[i] = False


    @tf.function
    def entropy_loss(self, softmax_confidences):
        entropy = - tf.math.xlogy(softmax_confidences, softmax_confidences)
        entropy = tf.math.reduce_sum(entropy, axis=0)

        # Bring the entropy to a term between 0 and 1.
        entropy /= tf.math.log(float(self.nvaes))

        # Heavily penalize entropies close to 0, since we want the information
        # to be shared.
        entropy = -tf.math.log(entropy)

        # Sum across all pixels of a given image.
        entropy = tf.math.reduce_sum(entropy, axis=[1, 2])

        # This now has shape (batch_size, 1).
        return entropy


    @tf.function
    def compute_loss(self, x):
        (softmax_confidences, vae_images) = self.model(x)

        loss_object = tf.keras.losses.MeanSquaredError()
        recall_loss = 0.0

        for i in range(self.nvaes):
            cur_loss = loss_object(x, vae_images[i], sample_weight=softmax_confidences[i])

            tf.summary.scalar(
                f'recall_loss_vae_{i}',
                cur_loss,
                step=None
            )

            recall_loss += cur_loss
        recall_loss /= self.nvaes
        recall_loss *= 28 * 28 * config.expand_per_width * config.expand_per_height

        kl_loss = 0.0
        for ivae, nvae in enumerate(self.vaes):
            cur_loss = VAE.compute_kl_loss(nvae.last_mean, nvae.last_logvar)
            tf.summary.scalar(f'kl-loss-vae-{ivae}',
                              tf.math.reduce_mean(config.beta * cur_loss),
                              step=None)
            kl_loss += cur_loss
        kl_loss /= self.nvaes

        ent_loss = self.entropy_loss(softmax_confidences)

        for i in range(self.nvaes):
            tf.summary.histogram(
                f'softmax_confidences_vae_{i}',
                softmax_confidences[i],
                step=None
            )

        tf.summary.scalar('ent_loss', tf.math.reduce_mean(config.gamma * ent_loss),
                          step=None)

        tf.summary.scalar('total_recall_loss', recall_loss,
                          step=None)

        vae_loss = tf.math.reduce_mean(
            recall_loss + config.beta * kl_loss + config.gamma * ent_loss)
        return vae_loss


    def get_trainable_variables(self):
        return self.model.trainable_variables


    def apply_gradients(self, vae_is_learning, grads_per_vae):
        for i in range(self.nvaes):
            grads = grads_per_vae[i]
            vars = self.vaes[i].get_trainable_variables()
            if vae_is_learning[i]:
                self.fast_optimizer.apply_gradients(zip(grads, vars))
            else:
                self.slow_optimizer.apply_gradients(zip(grads, vars))


    @tf.function
    def compute_gradients(self, X):
        variables = self.get_trainable_variables()
        with tf.GradientTape() as tape:
            loss = self.compute_loss(X)
        return tape.gradient(loss, variables), loss


    def fit(self, X, vae_is_learning):
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
        gradients, loss = self.compute_gradients(X)
        grads_per_vae = partition_gradients(gradients)

        for i in range(self.nvaes):
            for (grad, var) in zip(
                    grads_per_vae[i],
                    self.vaes[i].get_trainable_variables()):

                tf.summary.histogram(
                    f'grads_per_vae_{i}_weight_{var.name}',
                    grad,
                    step=None
                )

                tf.summary.histogram(
                    f'gvae_{i}_weight_{var.name}',
                    var,
                    step=None
                )

        self.apply_gradients(vae_is_learning, grads_per_vae)
        return loss


    def fit_on_dataset(self, D_train: tf.data.Dataset):
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
                    16 * config.batch_size
                ):
            loss = self.fit(X, self.vae_is_learning)
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

    @tf.function
    def run_on_input(self, X):
        (softmax_confidences, vae_images) = self.model(X)
        return (softmax_confidences, vae_images)


