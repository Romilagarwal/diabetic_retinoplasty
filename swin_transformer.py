import keras
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import os
import math
class PatchEmbed(layers.Layer):
    """Image to Patch Embedding"""
    def __init__(self, patch_size=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size,
                                  padding='valid', name='proj')
        self.norm = norm_layer(epsilon=1e-5, name='norm') if norm_layer else None
    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = self.proj(x)
        if self.norm:
            x = self.norm(x)
        return x
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size (int or tuple)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    if isinstance(window_size, int):
        window_h = window_w = window_size
    else:
        window_h, window_w = window_size
    x = tf.reshape(x, [B, H // window_h, window_h, W // window_w, window_w, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_h, window_w, C])
    return windows
def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: window size (int or tuple)
        H: Height of image
        W: Width of image
        C: Channels
    Returns:
        x: (B, H, W, C)
    """
    if isinstance(window_size, int):
        window_h = window_w = window_size
    else:
        window_h, window_w = window_size
    B = tf.shape(windows)[0] // (H * W // window_h // window_w)
    x = tf.reshape(windows, [B, H // window_h, W // window_w, window_h, window_w, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, C])
    return x
class WindowAttention(layers.Layer):
    """Window based multi-head self-attention module with relative position bias"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name='qkv')
        self.attn_drop = layers.Dropout(dropout_rate, name='attn_drop')
        self.proj = layers.Dense(dim, name='proj')
        self.proj_drop = layers.Dropout(dropout_rate, name='proj_drop')
        if isinstance(window_size, tuple):
            window_h, window_w = window_size
        else:
            window_h = window_w = window_size
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * window_h - 1) * (2 * window_w - 1), num_heads),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name='relative_position_bias_table')
        coords_h = tf.range(window_h)
        coords_w = tf.range(window_w)
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))
        coords = tf.reshape(coords, [2, -1])
        coords = tf.transpose(coords, [1, 0])
        coords_1 = tf.expand_dims(coords, 1)
        coords_2 = tf.expand_dims(coords, 0)
        relative_coords = coords_1 - coords_2
        relative_coords = tf.cast(relative_coords, tf.int32)
        relative_coords = relative_coords + tf.constant([window_h - 1, window_w - 1], dtype=tf.int32)
        relative_coords = relative_coords[..., 0] * (2 * window_w - 1) + relative_coords[..., 1]
        self.relative_position_index = tf.Variable(
            initial_value=tf.cast(relative_coords, tf.int32),
            trainable=False,
            name='relative_position_index')
    def call(self, x, mask=None, training=None):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        rel_pos_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, [-1]))
        if isinstance(self.window_size, tuple):
            window_h, window_w = self.window_size
            window_area = window_h * window_w
        else:
            window_area = self.window_size * self.window_size
        rel_pos_bias = tf.reshape(
            rel_pos_bias, [window_area, window_area, -1])
        rel_pos_bias = tf.transpose(rel_pos_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(rel_pos_bias, axis=0)
        if mask is not None:
            nW = tf.shape(mask)[0]
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N])
            attn = attn + mask
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x
class MLP(layers.Layer):
    """MLP module with dropout"""
    def __init__(self, hidden_features=None, out_features=None, dropout_rate=0., name=None):
        super().__init__(name=name)
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.fc1 = layers.Dense(hidden_features, name='fc1')
        self.act = layers.Activation('gelu')
        self.fc2 = layers.Dense(out_features if out_features is not None else hidden_features, name='fc2')
        self.drop = layers.Dropout(dropout_rate)
    def call(self, x, training=None):
        original_dims = tf.shape(x)
        original_shape = x.shape
        if len(original_shape) == 4:
            B, H, W, C = original_dims[0], original_dims[1], original_dims[2], original_dims[3]
            x = tf.reshape(x, [-1, C])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        if len(original_shape) == 4:
            x = tf.reshape(x, [B, H, W, -1])
        return x
class SwinTransformerBlock(layers.Layer):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, dropout_rate=0.,
                 attention_dropout_rate=0., trainable=True, dtype=None, name=None):
        super().__init__(name=name, trainable=trainable, dtype=dtype)  # Pass trainable and dtype to parent class
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # The rest of your implementation remains the same
        if isinstance(window_size, int):
            self.window_size_h = self.window_size_w = window_size
        else:
            self.window_size_h, self.window_size_w = window_size
            
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name='norm1')
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, dropout_rate=attention_dropout_rate, name='attn')
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name='norm2')
        self.mlp = MLP(
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            dropout_rate=dropout_rate,
            name='mlp'
        )
        
        # Determine shift size based on window size
        if isinstance(window_size, int):
            min_size = window_size
        else:
            min_size = min(window_size)
        if min_size > 0 and shift_size > 0:
            self.shift_size = min_size // 2
        else:
            self.shift_size = 0
    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = input_shape[1], input_shape[2]
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size_h),
                        slice(-self.window_size_h, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size_w),
                        slice(-self.window_size_w, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            if isinstance(self.window_size, int):
                window_size = self.window_size
            else:
                window_size = self.window_size_h
            mask_windows = window_partition(
                tf.convert_to_tensor(img_mask), window_size)
            if isinstance(self.window_size, tuple):
                window_h, window_w = self.window_size
                window_area = window_h * window_w
            else:
                window_area = self.window_size * self.window_size
            mask_windows = tf.reshape(
                mask_windows, [-1, window_area])
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                trainable=False,
                name='attn_mask')
        else:
            self.attn_mask = None
        super().build(input_shape)
    def call(self, x, training=None):
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        B = tf.shape(x)[0]
        shortcut = x
        x = self.norm1(x)
        x_reshaped = tf.reshape(x, [B, H, W, C])
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x_reshaped, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x_reshaped
        x_windows = window_partition(shifted_x, self.window_size)
        if isinstance(self.window_size, tuple):
            window_h, window_w = self.window_size
            window_area = window_h * window_w
        else:
            window_area = self.window_size * self.window_size
        x_windows = tf.reshape(
            x_windows, [-1, window_area, C])
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask, training=training)
        if isinstance(self.window_size, tuple):
            window_h, window_w = self.window_size
            attn_windows = tf.reshape(
                attn_windows, [-1, window_h, window_w, C])
        else:
            attn_windows = tf.reshape(
                attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = shortcut + x
        x = x + self.mlp(self.norm2(x), training=training)
        return x
class PatchMerging(layers.Layer):
    """Patch Merging Layer - downsamples by 2x"""
    def __init__(self, dim, norm_layer=layers.LayerNormalization, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False, name='reduction')
        self.norm = norm_layer(epsilon=1e-5, name='norm')
    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        assert H % 2 == 0 and W % 2 == 0, f"H and W ({H}, {W}) are not even."
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x
class BasicLayer(layers.Layer):
    """A basic Swin Transformer layer for one stage"""
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, dropout_rate=0.,
                 attention_dropout_rate=0., drop_path_rate=0.,
                 downsample=None, use_checkpoint=False, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate,
                    name=f'blocks_{i}'))
        if downsample is not None:
            self.downsample = downsample(dim=dim, name='downsample')
        else:
            self.downsample = None
    def call(self, x, training=None):
        for block in self.blocks:
            if self.use_checkpoint:
                x = tf.keras.utils.tf_utils.call_with_conditional_update(block, x, training=training)
            else:
                x = block(x, training=training)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
class SwinTransformer(Model):
    """Swin Transformer Model - Lightweight version for diabetic retinopathy"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=5,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 dropout_rate=0.1, attention_dropout_rate=0.1,
                 drop_path_rate=0.1, norm_layer=layers.LayerNormalization,
                 patch_norm=True, use_checkpoint=False, name="swin_transformer"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        self.layers = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                drop_path_rate=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                name=f'layers_{i_layer}'
            )
            self.layers.append(layer)
        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.head = layers.Dense(num_classes, name='head')
    def call(self, x, training=None):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, training=training)
        x = self.norm(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.head(x)
        return x
def create_swin_tiny_model(input_shape=(224, 224, 1), num_classes=5):
    """Create a memory-efficient Swin-T model for diabetic retinopathy"""
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=layers.LayerNormalization,
        patch_norm=True,
        use_checkpoint=False
    )
    outputs = model(x)
    return Model(inputs=inputs, outputs=outputs)
def create_hybrid_model(input_shape=(224, 224, 1), num_classes=5):
    """Create a hybrid EfficientNet+Swin model"""
    from tensorflow.keras.applications import EfficientNetB0
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    for layer in base_model.layers[:100]:
        layer.trainable = False
    features = base_model(x)
    feature_dim = features.shape[-1]
    swin_block = SwinTransformerBlock(
        dim=feature_dim,
        num_heads=8,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        name='swin_refine'
    )
    refined_features = swin_block(features)
    x = layers.GlobalAveragePooling2D()(refined_features)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("Memory growth enabled for GPU")
        except:
            print("Invalid device or cannot modify virtual devices once initialized")
    from tensorflow.keras.utils import plot_model
    import matplotlib.pyplot as plt
    model = create_hybrid_model(input_shape=(224, 224, 1), num_classes=5)
    model.summary()
    try:
        plot_model(model, to_file='hybrid_model.png', show_shapes=True)
        print("Model diagram saved to hybrid_model.png")
    except:
        print("Could not generate model diagram. Install pydot and graphviz for visualization.")
    test_batch = np.random.random((2, 224, 224, 1))
    with tf.device('/CPU:0'):  
        outputs = model.predict(test_batch)
    print(f"Output shape: {outputs.shape}")
    print("Memory test successful!")
