import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D,Lambda,GlobalAveragePooling3D, Multiply,LeakyReLU,LayerNormalization,MultiHeadAttention, Add,Input, Conv3D,Conv2D,Dropout, Concatenate, MaxPooling3D,MaxPooling2D, BatchNormalization, Activation, concatenate, Flatten, Dense, UpSampling3D,UpSampling2D, Conv3DTranspose,Activation,Reshape
from keras.models import Model
import keras.backend as K
import numpy as np
from tensorflow.keras import layers, Model,models
from tensorflow.keras.layers import GlobalAveragePooling3D, Reshape, Dense, Multiply
from tensorflow.keras import backend as K


class SEBlock(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        if len(input_shape) != 5:
            raise ValueError(f"Expected input shape to be 5D, but got {input_shape}")

        self.channels = input_shape[-1]

        self.global_avg_pool = layers.GlobalAveragePooling3D()
        self.global_max_pool = layers.GlobalMaxPooling3D()
        self.dense1 = layers.Dense(self.channels // self.ratio, activation='relu', use_bias=False)
        self.dense2 = layers.Dense(self.channels, activation='sigmoid', use_bias=False)


        self.multiply = layers.Multiply()

    def call(self, input_tensor):
        se = GlobalAveragePooling3D()(input_tensor)

        batch_size = tf.shape(input_tensor)[0]
        se = tf.expand_dims(tf.expand_dims(tf.expand_dims(se, 1), 1), 1)

        se = self.dense1(se)
        se = self.dense2(se)
        return Multiply()([input_tensor, se])

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({
            "ratio": self.ratio
        })
        return config

class spatial_attention_3d(layers.Layer):
    def __init__(self, **kwargs):
        super(spatial_attention_3d, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = layers.Conv3D(1, kernel_size=7, padding="same", activation="sigmoid")

    def call(self, input_feature):
        avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv(concat)
        return layers.Multiply()([input_feature, attention])

    def get_config(self):
        config = super().get_config()
        return config



class multi_scale_module(layers.Layer):
    def __init__(self, output_channel, **kwargs):
        super(multi_scale_module, self).__init__(**kwargs)
        self.output_channel = output_channel
        self.spatial_attention = spatial_attention_3d()
        self.channel_attention = SEBlock(ratio=1)

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError("The channel dimension of the inputs should be defined.")
        self.conv3 = layers.Conv3D(self.output_channel, kernel_size=3, padding="same", activation="relu")
        self.conv5 = layers.Conv3D(self.output_channel, kernel_size=5, padding="same", activation="relu")
        self.conv7 = layers.Conv3D(self.output_channel, kernel_size=7, padding="same", activation="relu")

    def call(self, input_feature):
        conv3 = self.conv3(input_feature)
        conv5 = self.conv5(input_feature)
        conv7 = self.conv7(input_feature)

        split_size = self.output_channel // 2
        conv3_1, conv3_2 = tf.split(conv3, num_or_size_splits=[split_size, split_size], axis=-1)
        conv5_1, conv5_2 = tf.split(conv5, num_or_size_splits=[split_size, split_size], axis=-1)
        conv7_1, conv7_2 = tf.split(conv7, num_or_size_splits=[split_size, split_size], axis=-1)

        conv3_attention_1 = self.spatial_attention(conv3_1)
        conv5_attention_1 = self.spatial_attention(conv5_1)
        conv7_attention_1 = self.spatial_attention(conv7_1)
        conv3_attention_2 = self.channel_attention(conv3_2)
        conv5_attention_2 = self.channel_attention(conv5_2)
        conv7_attention_2 = self.channel_attention(conv7_2)

        conv3_attention = Concatenate()([conv3_attention_1, conv3_attention_2])
        conv5_attention = Concatenate()([conv5_attention_1, conv5_attention_2])
        conv7_attention = Concatenate()([conv7_attention_1, conv7_attention_2])

        summed_output = Add()([conv3_attention, conv5_attention, conv7_attention])
        return summed_output

    def get_config(self):
        config = super(multi_scale_module, self).get_config()
        config.update({
            "output_channel": self.output_channel,
        })
        return config

class Transformer3DOptimized(layers.Layer):
    def __init__(self, d_model, num_heads=1, d_ff=64, window_size=(4, 4, 4), **kwargs):
        super(Transformer3DOptimized, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.window_size = window_size


        self.attention = None
        self.ffn = None
        self.dense = None

    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)
        self.dense = layers.Dense(self.d_model*64)
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.d_ff, activation='relu'),
            layers.Dense(self.d_model*64),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def split_to_windows(self, x, window_size):
        shape = tf.shape(x)
        batch_size, depth, height, width, channels = tf.unstack(shape)
        d, h, w = window_size
        depth_crop = depth // d * d
        height_crop = height // h * h
        width_crop = width // w * w
        x = x[:, :depth_crop, :height_crop, :width_crop, :]
        x = tf.reshape(
            x, [batch_size, depth_crop // d, d, height_crop // h, h, width_crop // w, w, channels]
        )
        x = tf.transpose(x, perm=[0, 1, 3, 5, 2, 4, 6, 7])  # (batch_size, wd, wh, ww, d, h, w, channels)
        return x

    def merge_windows(self, x, original_shape):
        shape = tf.shape(x)
        batch_size, wd, wh, ww, d, h, w, channels = tf.unstack(shape)
        _, depth, height, width, _ = original_shape
        x = tf.transpose(x, perm=[0, 1, 4, 2, 5, 3, 6, 7])  # (batch_size, wd, d, wh, h, ww, w, channels)
        x = tf.reshape(x, [batch_size, wd * d, wh * h, ww * w, channels])
        return x[:, :depth, :height, :width, :]


    def call(self, inputs, training=False):
        original_shape = tf.shape(inputs)
        original_shape = tf.unstack(original_shape)
        x = self.split_to_windows(inputs, self.window_size)
        shape = tf.shape(x)
        batch_size, wd, wh, ww, d, h, w, channels = tf.unstack(shape)
        x = tf.reshape(x, [-1, wd * wh * ww, d *h * w * channels])  # (batch_size * wd * wh * ww, window_size_prod, channels)
        x = self.dense(x)
        attn_output = self.attention(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # 前馈神经网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        channels = self.d_model
        out2 = tf.reshape(out2, [batch_size, wd, wh, ww, d, h, w, channels])
        outputs = self.merge_windows(out2, original_shape)
        outputs = tf.sigmoid(outputs)
        return outputs

    def get_config(self):
        config = super(Transformer3DOptimized, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "window_size": self.window_size,
        })
        return config

def spatial_attention_and_reduce(input_feature, name_prefix):
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True), name=f"{name_prefix}_avg_pool")(
        input_feature)  # (Batch, D, H, W, 1)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True), name=f"{name_prefix}_max_pool")(
        input_feature)  # (Batch, D, H, W, 1)

    concat = Concatenate(axis=-1, name=f"{name_prefix}_concat")([avg_pool, max_pool])  # (Batch, D, H, W, 2)

    spatial_attention_map = Conv3D(
        filters=1,
        kernel_size=(3, 3, 3),
        padding='same',
        activation='sigmoid',
        use_bias=False,
        name=f"{name_prefix}_spatial_conv"
    )(concat)

    attended_feature = Multiply()([input_feature, spatial_attention_map])  # (Batch, D, H, W, 8)

    reduced_output = GlobalAveragePooling3D(name=f"{name_prefix}_gap")(attended_feature)

    return reduced_output




def NEAMF_CAE_encoder(input_shape,batch_size):
    input_img = Input(shape=input_shape,batch_size=batch_size)
    imput_math = Input(shape=input_shape,batch_size=batch_size)

    x = multi_scale_module(output_channel=2)(input_img)
    m = Transformer3DOptimized(d_model=2, num_heads=4, d_ff=64,window_size=(4, 4, 4))(imput_math)
    x = Multiply()([x, m])
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    m = MaxPooling3D((2, 2, 2), padding='same')(m)
    x = multi_scale_module(output_channel=4)(x)
    m = Transformer3DOptimized(d_model=4, num_heads=4, d_ff=64,window_size=(4, 4, 4))(m)
    x = Multiply()([x, m])
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    m = MaxPooling3D((2, 2, 2), padding='same')(m)
    x = multi_scale_module(output_channel=8)(x)
    m = Transformer3DOptimized(d_model=8, num_heads=4, d_ff=64,window_size=(4, 4, 4))(m)
    x = Multiply()([x, m])
    encoded = MaxPooling3D((2, 2, 2), padding='same')(x)
    m = MaxPooling3D((2, 2, 2), padding='same')(m)

    x = multi_scale_module(output_channel=8)(encoded)
    m = Transformer3DOptimized(d_model=8, num_heads=4, d_ff=64,window_size=(4, 4, 4))(m)
    x = Multiply()([x, m])
    x = UpSampling3D((2, 2, 2))(x)
    m = UpSampling3D((2, 2, 2))(m)
    x = multi_scale_module(output_channel=4)(x)
    m = Transformer3DOptimized(d_model=4, num_heads=4, d_ff=64,window_size=(4, 4, 4))(m)
    x = Multiply()([x, m])
    x = UpSampling3D((2, 2, 2))(x)
    m = UpSampling3D((2, 2, 2))(m)
    x = multi_scale_module(output_channel=2)(x)
    m = Transformer3DOptimized(d_model=2, num_heads=4, d_ff=64,window_size=(4, 4, 4))(m)
    x = Multiply()([x, m])
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model([input_img, imput_math], decoded)
    return autoencoder


def NEAMF_CAE(input_shape, text_input_shape, batch_size=None):
    input_img = Input(shape=input_shape, batch_size=batch_size, name="input_img")
    input_math = Input(shape=input_shape, batch_size=batch_size, name="input_math")
    input_text = Input(shape=text_input_shape, name="input_text")

    input_img_1 = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 0], axis=-1))(input_img)
    input_img_2 = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 1], axis=-1))(input_img)
    input_math_1 = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 0], axis=-1))(input_math)
    input_math_2 = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 1], axis=-1))(input_math)

    combined_input_1 = Add()([input_img_1, input_math_1])
    x1 = multi_scale_module(output_channel=2, name="cam_1")(combined_input_1)
    m1 = Transformer3DOptimized(d_model=2, num_heads=4, d_ff=64, window_size=(4, 4, 4))(input_math_1)
    x1 = Multiply()([x1, m1])
    x1 = MaxPooling3D((2, 2, 2), padding='same')(x1)
    m1 = MaxPooling3D((2, 2, 2), padding='same')(m1)

    x1 = multi_scale_module(output_channel=4, name="cam_2")(x1)
    m1 = Transformer3DOptimized(d_model=4, num_heads=4, d_ff=64, window_size=(4, 4, 4))(m1)
    x1 = Multiply()([x1, m1])
    x1 = MaxPooling3D((2, 2, 2), padding='same')(x1)
    m1 = MaxPooling3D((2, 2, 2), padding='same')(m1)

    x1 = multi_scale_module(output_channel=8, name="cam_3")(x1)
    m1 = Transformer3DOptimized(d_model=8, num_heads=4, d_ff=64, window_size=(4, 4, 4))(m1)

    x1 = Multiply(name="cam_target_1")([x1, m1])
    x1 = MaxPooling3D((2, 2, 2), padding='same')(x1)

    encoded1 = spatial_attention_and_reduce(x1, name_prefix="branch1")

    # ================= 分支 2 =================
    combined_input_2 = Add()([input_img_2, input_math_2])
    x2 = multi_scale_module(output_channel=2)(combined_input_2)
    m2 = Transformer3DOptimized(d_model=2, num_heads=4, d_ff=64, window_size=(4, 4, 4))(input_math_2)
    x2 = Multiply()([x2, m2])
    x2 = MaxPooling3D((2, 2, 2), padding='same')(x2)
    m2 = MaxPooling3D((2, 2, 2), padding='same')(m2)

    x2 = multi_scale_module(output_channel=4)(x2)
    m2 = Transformer3DOptimized(d_model=4, num_heads=4, d_ff=64, window_size=(4, 4, 4))(m2)
    x2 = Multiply()([x2, m2])
    x2 = MaxPooling3D((2, 2, 2), padding='same')(x2)
    m2 = MaxPooling3D((2, 2, 2), padding='same')(m2)

    x2 = multi_scale_module(output_channel=8)(x2)
    m2 = Transformer3DOptimized(d_model=8, num_heads=4, d_ff=64, window_size=(4, 4, 4))(m2)

    x2 = Multiply(name="cam_target_2")([x2, m2])
    x2 = MaxPooling3D((2, 2, 2), padding='same')(x2)
    encoded2 = spatial_attention_and_reduce(x2, name_prefix="branch2")

    output = Concatenate()([encoded1, encoded2, input_text])

    output = Dense(16, activation='relu')(output)
    output = Dropout(0.3)(output)
    final_output = Dense(1, activation='sigmoid', name='classifier')(output)

    model = Model(inputs=[input_img, input_math, input_text], outputs=final_output)
    return model

