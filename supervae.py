import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

from vae import VAE


class SuperVAE(tf.keras.Model):
    def __init__(self, latent_dim: int) -> None:
        super(SuperVAE, self).__init__(name='SuperVAE')

        self.latent_dim = latent_dim

        self.nvaes = 10

        self.vaes = [
                VAE(latent_dim, 'VAE-{}'.format(i))
                for i in range(self.nvaes)]

    def encode(self, x):
        latent_vars = [
                vae.encode(x)
                for vae in self.vaes ]

        means = [lv[0] for lv in latent_vars]
        logvars = [lv[1] for lv in latent_vars]

        return (means, logvars)

    def reparametrize(self, means, logvars):
        z = []
        for i in range(self.nvaes):
            mean = means[i]
            logvar = logvars[i]
            vae = self.vaes[i]
            z.append(vae.reparametrize(mean, logvar))
        return z

    def decode(self, z):
        xs = []
        for i in range(self.nvaes):
            xs.append(self.vaes[i].decode(z))
        ret = tf.math.add_n(xs)
        return ret

    def compute_kl_loss(self, means, logvars):
        ret = []
        for i in range(self.nvaes):
            ret.append(self.vaes[i].compute_kl_loss(
                means[i], logvars[i]
                ))
        ret = tf.convert_to_tensor(ret)
        ret = tf.reduce_mean(ret, axis=0)
        return ret
