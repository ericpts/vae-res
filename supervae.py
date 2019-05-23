import tensorflow.keras as keras
import tensorflow as tf
import time
from config import global_config
import numpy as np

from vae import VAE


class SuperVAE(tf.keras.Model):

    def __init__(self, latent_dim: int, name: str) -> None:
        super(SuperVAE, self).__init__(name=name)

        self.latent_dim = latent_dim

        self.nvaes = global_config.nvaes

        inputs = keras.Input(
            shape=(
                global_config.img_height,
                global_config.img_width,
                global_config.img_channels
                )
            )

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

        self.set_lr_for_new_stage(1e-3)


    def set_lr_for_new_stage(self, lr: float):
        self.optimizer = tf.keras.optimizers.Adam(
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

        recall_loss_coef = global_config.img_width * global_config.img_height * global_config.img_channels

        for i in range(self.nvaes):
            cur_loss = loss_object(x, vae_images[i], sample_weight=softmax_confidences[i]) * recall_loss_coef
            tf.summary.scalar(
                f'recall_loss_vae_{i}',
                cur_loss,
                step=None
            )
            recall_loss += cur_loss
        recall_loss /= self.nvaes

        kl_loss = 0.0
        for ivae, nvae in enumerate(self.vaes):
            cur_loss = VAE.compute_kl_loss(nvae.last_mean, nvae.last_logvar)
            tf.summary.scalar(f'kl_loss_vae_{ivae}',
                              tf.math.reduce_mean(global_config.beta * cur_loss),
                              step=None)
            kl_loss += cur_loss
        kl_loss /= self.nvaes

        ent_loss = self.entropy_loss(softmax_confidences)

        if tf.summary.experimental.get_step() % 20 == 0:
            for i in range(self.nvaes):
                tf.summary.histogram(
                    f'softmax_confidences_vae_{i}',
                    softmax_confidences[i],
                    step=None
                )

        tf.summary.scalar('ent_loss', tf.math.reduce_mean(global_config.gamma * ent_loss),
                          step=None)

        tf.summary.scalar('total_recall_loss', recall_loss,
                          step=None)

        vae_loss = tf.math.reduce_mean(
            recall_loss + global_config.beta * kl_loss + global_config.gamma * ent_loss)

        tf.summary.scalar('total_loss', vae_loss,
                          step=None)
        return vae_loss


    def get_trainable_variables(self, vae_is_learning):
        ret = []
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
            self.optimizer.apply_gradients(
                zip(grads, variables))

        for D in D_train.batch(
                global_config.batch_size, drop_remainder=True).prefetch(
                    global_config.batch_size):
            t = time.time()
            X = D['img']
            loss = self.fit(X, variables, apply_gradients_fn)
            train_loss += loss * X.shape[0]
            train_size += X.shape[0]
            print(f'Done one batch in {time.time() - t} secs.')

        return train_loss / train_size


    def evaluate_on_dataset(self, D_test):
        test_loss = 0
        test_size = 0
        for D in D_test.batch(
                global_config.batch_size, drop_remainder=True):
            X = D['img']
            (softmax_confidences, vae_images) = self.model(X)

            y = D['bbox']
            y = np.swapaxes(y, 0, 1)

            # We might have labels for 10 objects, but only train on fewer.
            y = y[: global_config.nvaes]
            softmax_confidences = softmax_confidences[: global_config.nvaes]

            test_loss += -tf.math.xlogy(y, softmax_confidences)

            # test_loss += self.compute_loss(X) * X.shape[0]
            test_size += X.shape[0]

        return test_loss / test_size


    def run_on_input(self, X):
        (softmax_confidences, vae_images) = self.model(X)
        return (softmax_confidences, vae_images)


