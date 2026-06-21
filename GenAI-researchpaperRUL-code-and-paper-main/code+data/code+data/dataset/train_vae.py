import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# 1. Load Data for the final of the poeple of the
base_dir = r"d:\research paper - Copy\code+data\code+data\dataset\csv_format"
batteries = ["B0005", "B0006", "B0007", "B0018"]

print("Loading and aggregating dataset...")
cycle_data = []
for b in batteries:
    df = pd.read_csv(os.path.join(base_dir, f"{b}_discharge.csv"))
    # Aggregate per cycle to capture the degradation
    agg_df = df.groupby('id_cycle').agg({
        'Voltage_measured': ['max', 'min', 'mean'],
        'Current_measured': ['max', 'min', 'mean'],
        'Temperature_measured': 'mean',
        'Capacity': 'max' # Capacity is constant per cycle
    })
    agg_df.columns = [f"{c[0]}_{c[1]}" for c in agg_df.columns]
    agg_df['battery'] = b
    cycle_data.append(agg_df)

full_df = pd.concat(cycle_data, ignore_index=True)
features = full_df.drop('battery', axis=1).values

# Normalize features to [0, 1] for the VAE sigmoid output
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

print(f"Dataset shape: {features_scaled.shape}")

# 2. Build VAE Framework
class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=0.1, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta = beta

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # MSE reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.mse(data, reconstruction), axis=-1)
            )
            # KL Divergence loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Apply Beta parameter to bound the KL Divergence
            total_loss = reconstruction_loss + (self.beta * kl_loss)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

input_dim = features_scaled.shape[1]
latent_dim = 2 # 2D latent space for battery state representation

# Encoder
encoder_inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(16, activation="relu")(encoder_inputs)
x = layers.Dense(8, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8, activation="relu")(latent_inputs)
x = layers.Dense(16, activation="relu")(x)
decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Callback to dynamically control KL divergence constraint (< 0.5)
class KLConstraintCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        kl = logs.get('kl_loss')
        # If KL divergence exceeds 0.45, dynamically increase beta penalty aggressively
        if kl > 0.45:
            self.model.beta += 2.0
            print(f" -> High KL ({kl:.4f}): Increased Beta to {self.model.beta.numpy():.2f}")
        elif kl < 0.2 and self.model.beta > 0.5:
            self.model.beta -= 0.5

# Train VAE
vae = VAE(encoder, decoder, beta=10.0) # Initial beta 10.0
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005))

print("\n--- Training Beta-VAE Model ---")
# Train for adequate epochs to converge
history = vae.fit(features_scaled, epochs=150, batch_size=16, callbacks=[KLConstraintCallback()], verbose=2)

final_kl_div = vae.kl_loss_tracker.result().numpy()
print(f"\nTraining Complete. Final KL Divergence: {final_kl_div:.4f}")
if final_kl_div < 0.5:
    print("SUCCESS: KL Divergence securely below 0.5 threshold.")
else:
    print("WARNING: KL Divergence exceeded 0.5 threshold.")

# 3. Generate Synthetic Data
print("\n--- Synthesizing Data ---")
num_samples_to_generate = 2000 # Generate a diverse augmented dataset
random_latent_vectors = tf.random.normal(shape=(num_samples_to_generate, latent_dim))
synthetic_scaled = vae.decoder(random_latent_vectors).numpy()

# Inverse transform back to original feature scale
synthetic_features = scaler.inverse_transform(synthetic_scaled)
synthetic_df = pd.DataFrame(synthetic_features, columns=full_df.drop('battery', axis=1).columns)

out_file = os.path.join(base_dir, "synthetic_battery_cycles_vae.csv")
synthetic_df.to_csv(out_file, index=False)
print(f"Successfully generated {num_samples_to_generate} synthetic cycles.")
print(f"Saved to: {out_file}")
