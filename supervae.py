import tensorflow.keras as keras
import tensorflow as tf
from config import global_config
import numpy as np

from unet import UNet
from vae import VAE


class SuperVAE(tf.keras.Model):

    def __init__(self, latent_dim: int, name: str) -> None:
        super(SuperVAE, self).__init__(name=name)

        self.latent_dim = latent_dim

        self.nvaes = global_config.nvaes

        self.vaes = [
            VAE(latent_dim, 'VAE-{}'.format(i)) for i in range(self.nvaes)
        ]

        self.unet = UNet(global_config.n_unet_blocks)

        self.vae_is_learning = np.array([True for i in range(self.nvaes)])


    def set_lr_for_new_stage(self, lr: float):
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=lr,
            epsilon=0.1,
        )

    def freeze_all(self):
        for i in range(self.nvaes):
            self.freeze_vae(i)

    def unfreeze_all(self):
        for i in range(self.nvaes):
            self.unfreeze_vae(i)

    def unfreeze_vae(self, i: int):
        assert 0 <= i and i < self.nvaes
        self.vae_is_learning[i] = True

    def freeze_vae(self, i: int):
        assert 0 <= i and i < self.nvaes
        self.vae_is_learning[i] = False


    @tf.function
    def compute_loss(self, X):
        (log_residual, vae_images, vae_masks, masks) = self.run_on_input(X)

        loss_object = tf.keras.losses.MeanSquaredError()
        recall_loss = 0.0
        recall_loss_coef = 28 * 28 * global_config.expand_per_width * global_config.expand_per_height

        for i in range(self.nvaes):
            cur_loss = loss_object(
                X, vae_images[i],
                sample_weight=masks[i]) * recall_loss_coef

            tf.summary.scalar(f'recall_loss_vae_{i}', cur_loss, step=None)
            recall_loss += cur_loss
        recall_loss /= self.nvaes

        kl_loss = 0.0
        for ivae, nvae in enumerate(self.vaes):
            cur_loss = VAE.compute_kl_loss(nvae.last_mean, nvae.last_logvar)
            tf.summary.scalar(
                f'raw_kl_loss_vae_{ivae}',
                tf.math.reduce_mean(global_config.beta * cur_loss),
                step=None)
            kl_loss += cur_loss
        kl_loss /= self.nvaes

        # TODO maybe these are useful in the future.
        #
        # if tf.summary.experimental.get_step() % 20 == 0:
        #     for i in range(self.nvaes):
        #         tf.summary.histogram(
        #             f'softmax_confidences_vae_{i}',
        #             softmax_confidences[i],
        #             step=None)

        ent_loss = 0
        for i in range(self.nvaes):
            # TODO: Maybe make this into a KL divergence instead.
            cur_loss = loss_object(
                masks[i], vae_masks[i]
            )
            tf.summary.scalar(f'mask_loss_vae_{i}', cur_loss, step=None)
            ent_loss += cur_loss
        ent_loss /= self.nvaes

        tf.summary.scalar('total_recall_loss', recall_loss, step=None)

        total_loss = tf.math.reduce_mean(recall_loss +
                                       global_config.beta * kl_loss +
                                       global_config.gamma * ent_loss)

        tf.summary.scalar('total_loss', total_loss, step=None)
        return total_loss

    def get_trainable_variables(self, vae_is_learning):
        ret = [
            *self.unet.trainable_variables
        ]
        for i in range(self.nvaes):
            if vae_is_learning[i]:
                ret.extend(self.vaes[i].get_trainable_variables())
        return ret

    @tf.function
    def compute_gradients(self, X, variables):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(X)
        return tape.gradient(loss, variables), loss

    def fit(self, X, variables, apply_gradients_fn):
        gradients, loss = self.compute_gradients(X, variables)
        apply_gradients_fn(gradients)
        return loss

    def fit_on_dataset(self, D_train: tf.data.Dataset):
        train_loss = 0
        train_size = 0

        variables = self.get_trainable_variables(self.vae_is_learning)

        @tf.function
        def apply_gradients_fn(grads):
            self.optimizer.apply_gradients(zip(grads, variables))

        for (X, y) in D_train.batch(
                global_config.batch_size,
                drop_remainder=True).prefetch(16 * global_config.batch_size):
            loss = self.fit(X, variables, apply_gradients_fn)
            train_loss += loss * X.shape[0]
            train_size += X.shape[0]

        return train_loss / train_size

    def evaluate_on_dataset(self, D_test):
        test_loss = 0
        test_size = 0
        for (X, y) in D_test.batch(
                global_config.batch_size * 8, drop_remainder=True):
            test_loss += self.compute_loss(X) * X.shape[0]
            test_size += X.shape[0]

        return test_loss / test_size

    def run_on_input(self, X):
        log_residual = tf.zeros_like(X)

        vae_images = []
        vae_masks = []
        masks = []

        for i in range(self.nvaes):
            log_m, log_1minusm = self.unet(
                tf.concat([X, log_residual], axis=-1)
            )

            mask = tf.math.add(log_residual, log_m)

            (rimage, rmask) = self.vaes[i](
                tf.concat([X, mask], axis=-1)
            )

            log_residual = tf.math.add(log_residual, log_1minusm)

            vae_images.append(rimage)
            vae_masks.append(rmask)
            masks.append(mask)

        vae_masks = tf.split(
            tf.math.softmax(
                tf.concat(vae_masks, axis=-1),
                axis=-1
            ),
            num_or_size_splits=self.nvaes,
            axis=-1,
        )

        return (
            log_residual,
            vae_images,
            vae_masks,
            masks
        )
