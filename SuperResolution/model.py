import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="TrainModel")
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=channels)
        self.ln = tf.keras.layers.LayerNormalization()
    def call(self, x):
        x = tf.reshape(x, [-1, self.size * self.size, self.channels])
        x_ln = self.ln(x)
        attention_value = self.mha(query=x_ln, key=x_ln, value=x_ln)
        attention_value = attention_value + x
        attention_value = tf.reshape(attention_value, [-1, self.size, self.size, self.channels])
        return attention_value
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "size": self.size
        })
        return config

@keras.saving.register_keras_serializable(package="TrainModel")
class Upsample(tf.keras.layers.Layer):
    def __init__(self, outc):
        super(Upsample, self).__init__()
        self.outc = outc
        self.up = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')
        self.conv = tf.keras.layers.Conv2D(outc, kernel_size=3, padding='same', use_bias=False)
    def call(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
    def get_config(self):
        config = super().get_config()
        config.update({
            "outc": self.outc
        })
        return config

@keras.saving.register_keras_serializable(package="TrainModel")
class Downsample(tf.keras.layers.Layer):
    def __init__(self, outc):
        super(Downsample, self).__init__()
        self.outc = outc
        # Alternative: Reduce using Conv2D
        self.down = tf.keras.layers.MaxPooling2D(2)
        self.conv = tf.keras.layers.Conv2D(outc, kernel_size=3, padding='same', use_bias=False)
    def call(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x
    def get_config(self):
        config = super().get_config()
        config.update({
            "outc": self.outc
        })
        return config

@keras.saving.register_keras_serializable(package="TrainModel")
class Block(tf.keras.layers.Layer):
    def __init__(self, outc):
        super(Block, self).__init__()
        self.outc = outc
        self.block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(outc, kernel_size=3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(axis=-1), # or GroupNormalization
            tf.keras.layers.Activation('relu') # or Swish
        ])
    def call(self, x):
        x = self.block(x)
        return x
    def get_config(self):
        config = super().get_config()
        config.update({
            "outc": self.outc
        })
        return config

@keras.saving.register_keras_serializable(package="TrainModel")
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, outc):
        super(ResnetBlock, self).__init__()
        self.outc = outc
        self.block1 = Block(outc)
        self.block2 = Block(outc)
        self.emb = tf.keras.Sequential([
            tf.keras.layers.Activation('silu'),
            tf.keras.layers.Dense(outc)
        ])
    def call(self, x, t):
        """
        Alternativa:
        x_b = x + emb
        x_b = self.block1(x_b)
        x_b = x_b + emb
        x_b = self.block2(x)
        return x_b + self.res_conv(x)
        """
        x_b = self.block1(x)
        emb = tf.tile(self.emb(t)[:, None, None, :], [1, x.shape[-3], x.shape[-2], 1])
        x_b = x_b + emb
        x_b = self.block2(x_b)
        return x_b
    def build(self, input_shape):
        super().build(input_shape)
    def get_config(self):
        config = super().get_config()
        config.update({
            "outc": self.outc
        })
        return config
    
@keras.saving.register_keras_serializable(package="TrainModel")
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def call(self, noise_level):
        count = self.dim // 2
        step = tf.range(count, dtype='float32') / count
        arg1 = tf.cast(-tf.math.log(1e4), dtype='float32')
        arg2 = tf.cast(tf.expand_dims(step, axis=0), dtype='float32')
        encoding = tf.cast(tf.expand_dims(noise_level, axis=1), dtype='float32') * tf.cast(tf.math.exp(arg1 * arg2), dtype='float32')
        encoding = tf.concat([tf.math.sin(encoding), tf.math.cos(encoding)], axis=-1)
        return tf.squeeze(encoding, axis=1)
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim
        })
        return config

@keras.saving.register_keras_serializable(package="TrainModel")
class DiffusionModel(tf.keras.Model):
    def __init__(self, 
                 in_ch=3, 
                 out_ch=3, 
                 inner_channel=128, 
                 ch_mult=(1, 2, 4, 4, 8, 8), 
                 bot_blocks=2, 
                 image_size=256, 
                 noise_steps=1000, 
                 pred_steps=1000, 
                 beta_start=1e-4, 
                 beta_end=0.02, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.inner_channel = inner_channel
        self.ch_mult = ch_mult
        self.bot_blocks = bot_blocks
        
        num_mults = len(ch_mult)
        self.noise_encoder = tf.keras.Sequential([PositionalEncoding(inner_channel)])
        self.in_conv = tf.keras.layers.Conv2D(inner_channel, kernel_size=3, padding='same', use_bias=False)

        self.down_layers = []
        self.down_resnet = []

        for i in range(1, num_mults, 1):
            self.down_layers.append(Downsample(inner_channel * ch_mult[i]))
            self.down_resnet.append(ResnetBlock(inner_channel * ch_mult[i]))
        
        self.self_att1 = SelfAttention(inner_channel * ch_mult[-1], image_size // (2 ** (num_mults - 1)))
        self.bot_layers = []

        for i in range(bot_blocks):
            self.bot_layers.append(ResnetBlock(inner_channel * ch_mult[-1]))
        
        self.up_layers = []
        self.up_resnet = []

        for i in range(num_mults - 2, -1, -1):
            self.up_layers.append(Upsample(inner_channel * ch_mult[i]))
            self.up_resnet.append(ResnetBlock(inner_channel * ch_mult[i]))
        
        self.out_conv = tf.keras.layers.Conv2D(3, kernel_size=3, padding='same')

        # Params
        self.noise_steps = 1000
        self.img_size = image_size
        self.pred_nsteps = 200
        self.beta_start = 1e-4
        self.beta_end = 0.02

        self.beta = tf.linspace(self.beta_start, self.beta_end, self.noise_steps)
        self.alpha = 1.0 - self.beta
        self.gamma = tf.math.cumprod(self.alpha, axis=0)

        self.noise_rng = tf.random.Generator.from_non_deterministic_state()
    
    def call(self, x, t):
        t = self.noise_encoder(t)
        residuals = [self.in_conv(x)]
        for down, resnet in zip(self.down_layers, self.down_resnet):
            residuals.append(resnet(down(residuals[-1]), t))
        
        x = residuals.pop()
        x = self.self_att1(x)
        for bot in self.bot_layers:
            x = bot(x, t)
        
        for up, resnet in zip(self.up_layers, self.up_resnet):
            x = resnet(tf.concat([up(x), residuals.pop()], axis=-1), t)
        
        x = self.out_conv(x)
        return x
    
    def noise_inputs(self, x, s, u_scale, epsilon):
        l_a, l_b = tf.gather(self.gamma, s - 1), tf.gather(self.gamma, s)
        noise_scale = l_a + u_scale * (l_b - l_a)
        sqrt_noise_scale = tf.math.sqrt(noise_scale)[:, :, None, None]
        sqrt_one_minus_noise_scale = tf.math.sqrt(1.0 - noise_scale)[:, :, None, None]
        return sqrt_noise_scale * x + sqrt_one_minus_noise_scale * epsilon
    
    def train_step(self, data):
        y, rnd = data
        s, u_scale, noise = rnd
        y_t = self.noise_inputs(y, s, u_scale, noise)
        with tf.GradientTape() as tape:
            predicted_noise = self(y_t, s, training=True)
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
        y_t = data
        for i in tf.range(self.pred_nsteps - 1, 0, -1, dtype='int32'):
            t = tf.cast(tf.ones(y_t.shape[0]), 'int32') * i
            predicted_noise = self(y_t, t, training=False)
            alpha = tf.gather(self.alpha, t)[:, None, None, None]
            gamma = tf.gather(self.gamma, t)[:, None, None, None]
            if i > 1:
                noise = self.noise_rng.normal(shape=y_t.shape, mean=0.0, stddev=1.0, dtype='float32')
            else:
                noise = tf.zeros_like(y_t)
            y_t = (1 / tf.math.sqrt(alpha)) * (y_t - ((1.0 - alpha) / (tf.math.sqrt(1.0 - gamma))) * predicted_noise) + (tf.math.sqrt(1.0 - alpha) * noise)
        return y_t
    
    def build(self, input_shape):
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "inner_channel": self.inner_channel,
            "ch_mult": self.ch_mult,
            "bot_blocks": self.bot_blocks,
            "image_size": self.img_size,
            "noise_steps": self.noise_steps,
            "pred_steps": self.pred_nsteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        config["in_ch"] = keras.saving.deserialize_keras_object(config["in_ch"])
        config["out_ch"] = keras.saving.deserialize_keras_object(config["out_ch"])
        config["inner_channel"] = keras.saving.deserialize_keras_object(config["inner_channel"])
        config["ch_mult"] = keras.saving.deserialize_keras_object(config["ch_mult"])
        config["bot_blocks"] = keras.saving.deserialize_keras_object(config["bot_blocks"])
        config["image_size"] = keras.saving.deserialize_keras_object(config["image_size"])
        
        config["noise_steps"] = keras.saving.deserialize_keras_object(config["noise_steps"])
        config["pred_steps"] = keras.saving.deserialize_keras_object(config["pred_steps"])
        config["beta_start"] = keras.saving.deserialize_keras_object(config["beta_start"])
        config["beta_end"] = keras.saving.deserialize_keras_object(config["beta_end"])

        return cls(**config)
    
    def get_build_config(self):
        build_config = super().get_build_config()
        return build_config
    
    def build_from_config(self, config):
        self.build(config["input_shape"])

rng = tf.random.Generator.from_non_deterministic_state()
normal = rng.normal(shape=[9, 256, 256, 3], mean=0.0, stddev=1.0, dtype='float32')
t_sample = rng.uniform(shape=[9, 1], minval=1, maxval=1000, dtype='int32')

# Uncomment below to test model

#model = DiffusionModel(inner_channel=64, ch_mult=(1, 2, 4, 8), bot_blocks=3, image_size=64)#UNet()
#model(normal, t_sample)
#model.summary()