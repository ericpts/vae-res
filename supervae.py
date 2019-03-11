import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from config import *

from vae import VAE


class SuperVAE(tf.keras.Model):
    def __init__(self, latent_dim: int, name: str) -> None:
        super(SuperVAE, self).__init__(name=name)

        self.latent_dim = latent_dim

        self.nvaes = config.nvaes

        inputs = keras.Input(
                shape=(28 * config.expand_per_height, 28 * config.expand_per_width, 1))

        self.vaes = [
                VAE(latent_dim, 'VAE-{}'.format(i))
                for i in range(self.nvaes)]

        self.vae_status = {
                i: True for i in range(self.nvaes)
        }

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
                inputs=inputs,
                outputs=[softmax_confidences, vae_images]
                )


    def unfreeze_vae(self, i: int):
        assert i in self.vae_status
        self.vae_status[i] = True


    def freeze_vae(self, i: int):
        assert i in self.vae_status
        self.vae_status[i] = False


    def compute_loss(self, x):
        (softmax_confidences, vae_images) = self.model(x)

        loss_object = tf.keras.losses.MeanSquaredError()
        recall_loss = 0
        for i in range(self.nvaes):
            recall_loss += loss_object(x, vae_images[i], sample_weight=softmax_confidences[i])
        recall_loss /= self.nvaes
        recall_loss *= 28 * 28 * config.expand_per_width * config.expand_per_height

        kl_loss = 0
        for nvae in self.vaes:
            kl_loss += VAE.compute_kl_loss(nvae.last_mean, nvae.last_logvar)
        kl_loss /= self.nvaes

        vae_loss = tf.math.reduce_mean(recall_loss + config.beta * kl_loss)
        return vae_loss


    def get_trainable_variables(self):
        ret = []
        for i in range(self.nvaes):
            if self.vae_status[i]:
                ret.extend(self.vaes[i].get_trainable_variables())
        return ret


    def sample(self, eps):
        vae_images = []
        vae_confidences = []

        for vae in self.vaes:
            (image, confidence) = vae.decode(eps)
            vae_images.append(image)
            vae_confidences.append(confidence)

        vae_confidences = tf.convert_to_tensor(vae_confidences)
        vae_images = tf.convert_to_tensor(vae_images)

        softmax_confidences = tf.keras.activations.softmax(vae_confidences, axis=0)

        return tf.math.reduce_sum(
                vae_images * softmax_confidences,
                axis=0
                )


    def summarize(self):
        self.vaes[0].summarize()
