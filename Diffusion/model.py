import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="TrainModel")
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels, size, prefix):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=channels, name=f'{prefix}_SA_mha')
        self.ln = tf.keras.layers.LayerNormalization()
        self.ff_self = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(name=f'{prefix}_SA_layer_norm'),
            tf.keras.layers.Dense(channels, name=f'{prefix}_SA_dense_1'),
            tf.keras.layers.Activation('gelu', name=f'{prefix}_SA_gelu_act'),
            tf.keras.layers.Dense(channels, name=f'{prefix}_SA_dense_2')
        ], name=f'{prefix}_SA_sequential')
    def call(self, x):
        x = tf.reshape(x, [-1, self.size * self.size, self.channels])
        
        x_ln = self.ln(x)
        attention_value = self.mha(query=x_ln, key=x_ln, value=x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # The reshaping should be of [attention_value] not [x]
        # For some reason, even with this mistake it produces good results
        attention_value = tf.reshape(x, [-1, self.size, self.size, self.channels])
        return attention_value

@keras.saving.register_keras_serializable(package="TrainModel")
class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, mid_channels=None, prefix='', residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(mid_channels, kernel_size=3, padding='same', use_bias=False, name=f'{prefix}_DC_conv_1'), # False
            tf.keras.layers.GroupNormalization(groups=1, axis=-1, name=f'{prefix}_DC_group_norm_1'),#, epsilon=1e-5),
            tf.keras.layers.Activation('gelu', name=f'{prefix}_DC_gelu_act'),
            tf.keras.layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False, name=f'{prefix}_DC_conv_2'), # False
            tf.keras.layers.GroupNormalization(groups=1, axis=-1, name=f'{prefix}_DC_group_norm_2')#, epsilon=1e-5)
        ], name=f'{prefix}_DC_sequential')
    def call(self, x):
        if self.residual:
            return tf.nn.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

@keras.saving.register_keras_serializable(package="TrainModel")
class Down(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, prefix, emb_dim=256):
        super().__init__()
        self.maxpool_conv = tf.keras.Sequential([
            tf.keras.layers.MaxPooling2D(2, name=f'{prefix}_DN_max_pool'),
            DoubleConv(in_channels, in_channels, residual=True, prefix=f'{prefix}_Down_1_'),
            DoubleConv(in_channels, out_channels, prefix=f'{prefix}_Down_2_'),
        ], name=f'{prefix}_DN_sequential_1')
        self.emb_layer = tf.keras.Sequential([
            tf.keras.layers.Activation('silu', name=f'{prefix}_DN_silu_act'),
            tf.keras.layers.Dense(out_channels, name=f'{prefix}_DN_dense')
        ], name=f'{prefix}_DN_sequential_2')
    def call(self, x, t):
        x = self.maxpool_conv(x)
        emb = tf.tile(self.emb_layer(t)[:, None, None, :], [1, x.shape[-3], x.shape[-2], 1])
        return x + emb
        #return x

@keras.saving.register_keras_serializable(package="TrainModel")
class Up(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, prefix, emb_dim=256):
        super().__init__()
        # Upsampling may be improved with align corners, like in Pytorch
        self.up = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name=f'{prefix}_UP_upsample')
        self.conv = tf.keras.Sequential([
            DoubleConv(in_channels, in_channels, residual=True, prefix=f'{prefix}_Up_1_'),
            DoubleConv(in_channels, out_channels, in_channels // 2, prefix=f'{prefix}_Up_2_')
        ], name=f'{prefix}_UP_sequential_1')
        self.emb_layer = tf.keras.Sequential([
            tf.keras.layers.Activation('silu', name=f'{prefix}_UP_silu_act'),
            tf.keras.layers.Dense(out_channels, name=f'{prefix}_UP_dense')
        ], name=f'{prefix}_UP_sequential_2')
    def call(self, x, skip_x, t):
        x = self.up(x)
        x = tf.concat([skip_x, x], axis=-1)
        x = self.conv(x)
        emb = tf.tile(self.emb_layer(t)[:, None, None, :], [1, x.shape[-3], x.shape[-2], 1])
        return x + emb

@keras.saving.register_keras_serializable(package="TrainModel")
class DiffusionTraining(tf.keras.Model):
    def set_diffusion_args(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64):
        self.noise_layer = tf.keras.layers.GaussianNoise(stddev=1.0, name='DT_gauss_noise')
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        
        self.beta = tf.linspace(self.beta_start, self.beta_end, self.noise_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0)
        
    def noise_images(self, x, t, training=True):
        sqrt_alpha_hat = tf.math.sqrt(tf.gather(self.alpha_hat, t))[:, None, None, None]
        sqrt_one_minus_alpha_hat = tf.math.sqrt(1 - tf.gather(self.alpha_hat, t))[:, None, None, None]
        epsilon = self.noise_layer(x, training=training)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    @tf.function
    def train_step(self, data):
        x, t = data
        t = tf.transpose(t)[0]
        x, noise = self.noise_images(x, t)
        with tf.GradientTape() as tape:
            predicted_noise = self((x, t), training=True)
            loss_value = self.compute_loss(y=noise, y_pred=predicted_noise)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss_value)
            else:
                metric.update_state(x, predicted_noise)
        return {m.name: m.result() for m in self.metrics}
    @tf.function
    def predict_step(self, data):
        for i in tf.range(self.noise_steps - 1, 0, -1, dtype='int32'):
            t = tf.cast(tf.ones([data.shape[0]]), dtype='int32') * i
            predicted_noise = self((data, t), training=False)
            alpha = tf.gather(self.alpha, t)[:, None, None, None]
            alpha_hat = tf.gather(self.alpha_hat, t)[:, None, None, None]
            beta = tf.gather(self.beta, t)[:, None, None, None]
            if i > 1:
                noise = self.noise_layer(tf.ones([data.shape[0], self.img_size, self.img_size, 3]), training=True)
            else:
                noise = tf.zeros_like(data)
            data = 1 / tf.math.sqrt(alpha) * (data - ((1 - alpha) / (tf.math.sqrt(1 - alpha_hat))) * predicted_noise) + tf.math.sqrt(beta) * noise
        return data
    
    def sample_step(self, x, n=3):
        rng = tf.random.Generator.from_non_deterministic_state()
        ts = rng.uniform(shape=[n], minval=1, maxval=self.noise_steps, dtype='int32')
        
        x_t, noise = self.noise_images(x, ts)
        prediction = self((x_t, ts), training=False)
        return prediction, noise, x_t

def pos_encoding(t, channels):
    inv_freq = 1.0 / (
        10000
        ** (tf.range(0, channels, 2, dtype='float32') / channels)
    )
    pos_enc_a = tf.math.sin(tf.tile(t, [1, channels // 2]) * inv_freq)
    pos_enc_b = tf.math.cos(tf.tile(t, [1, channels // 2]) * inv_freq)
    pos_enc = tf.concat([pos_enc_a, pos_enc_b], axis=-1)
    return pos_enc

@keras.saving.register_keras_serializable(package="TrainModel")
class PosEncLayer(tf.keras.layers.Layer):
    #def __init__(self):
    #    super(PosEncLayer, self).__init__()
    
    # The commented line below is in case of [t] input shape
    # has an extra dimension. To be fixed for more clarity.
    def call(self, inputs, time_dim=256):
        #inputs = tf.squeeze(inputs, axis=0)
        t = tf.cast(tf.expand_dims(inputs, -1), dtype='float32')
        t = pos_encoding(t, time_dim)
        return t
    
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (tf.range(0, channels, 2, dtype='float32') / channels)
        )
        pos_enc_a = tf.math.sin(tf.tile(t, [1, channels // 2]) * inv_freq)
        pos_enc_b = tf.math.cos(tf.tile(t, [1, channels // 2]) * inv_freq)
        pos_enc = tf.concat([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc

class DifussionModel(tf.keras.Model):
    def __init__(self, x_shape, t_shape, c_in=3, c_out=3, time_dim=256, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.time_dim = time_dim

        self.pos_enc_layer = PosEncLayer()
        
        self.inc = DoubleConv(c_in, 64, prefix="inc")
        self.down1 = Down(64, 128, prefix="down1")
        self.sa1 = SelfAttention(128, 32, prefix="self_att_1")
        self.down2 = Down(128, 256, prefix="down2")
        self.sa2 = SelfAttention(256, 16, prefix="self_att_2")
        self.down3 = Down(256, 256, prefix="down3")
        self.sa3 = SelfAttention(256, 8, prefix="self_att_3")

        self.bot1 = DoubleConv(256, 512, prefix="bot1")
        self.bot2 = DoubleConv(512, 512, prefix="bot2")
        self.bot3 = DoubleConv(512, 256, prefix="bot3")

        self.up1 = Up(512, 128, prefix="up1")
        self.sa4 = SelfAttention(128, 16, prefix="self_att_4")
        self.up2 = Up(256, 64, prefix="up2")
        self.sa5 = SelfAttention(64, 32, prefix="self_att_5")
        self.up3 = Up(128, 64, prefix="up3")
        self.sa6 = SelfAttention(64, 64, prefix="self_att_6")
        self.outc = tf.keras.layers.Conv2D(c_out, kernel_size=1, name="outc")

        # Diffusion args
        self.noise_layer = tf.keras.layers.GaussianNoise(stddev=1.0)
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        self.beta = tf.linspace(self.beta_start, self.beta_end, self.noise_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0)
        self.noise_rng = tf.random.Generator.from_non_deterministic_state()
    
    def call(self, x, t):
        t = self.pos_enc_layer(t, time_dim=self.time_dim)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        x = self.outc(x)

        return x

    def noise_images(self, x, t, epsilon):
        sqrt_alpha_hat = tf.math.sqrt(tf.gather(self.alpha_hat, t))[:, None, None, None]
        sqrt_one_minus_alpha_hat = tf.math.sqrt(1 - tf.gather(self.alpha_hat, t))[:, None, None, None]
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon

    def train_step(self, data):
        x, rnd = data
        t, noise = rnd
        t = tf.transpose(t)[0]
        x_t = self.noise_images(x, t, noise)
        with tf.GradientTape() as tape:
            predicted_noise = self(x_t, t, training=True)
            loss_value = self.compute_loss(y=noise, y_pred=predicted_noise)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss_value)
            else:
                metric.update_state(noise, predicted_noise)
        return {m.name: m.result() for m in self.metrics}
    @tf.function
    def predict_step(self, data):
        for i in tf.range(self.noise_steps - 1, 0, -1, dtype='int32'):
            t = tf.cast(tf.ones(data.shape[0]), "int32") * i
            predicted_noise = self(data, t, training=False)
            alpha = tf.gather(self.alpha, t)[:, None, None, None]
            alpha_hat = tf.gather(self.alpha_hat, t)[:, None, None, None]
            beta = tf.gather(self.beta, t)[:, None, None, None]
            if i > 1:
                noise = self.noise_rng.normal(shape=data.shape, mean=0.0, stddev=1.0, dtype='float32')
            else:
                noise = tf.zeros_like(data)
            data = (1 / tf.math.sqrt(alpha)) * (data - ((1 - alpha) / (tf.math.sqrt(1 - alpha_hat))) * predicted_noise) + (tf.math.sqrt(beta) * noise)
        return data