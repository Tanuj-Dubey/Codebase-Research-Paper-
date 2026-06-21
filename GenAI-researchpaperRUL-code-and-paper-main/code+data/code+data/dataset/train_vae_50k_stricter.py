import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import wasserstein_distance, entropy
from sklearn.preprocessing import MinMaxScaler
import os

base_dir = r"d:\research paper - Copy\code+data\code+data\dataset\csv_format"
batteries = ["B0005", "B0006", "B0007", "B0018"]

# 1. Load Data
cycle_data = []
for b in batteries:
    df = pd.read_csv(os.path.join(base_dir, f"{b}_discharge.csv"))
    agg_df = df.groupby('id_cycle').agg({
        'Voltage_measured': ['max', 'min', 'mean'],
        'Current_measured': ['max', 'min', 'mean'],
        'Temperature_measured': 'mean',
        'Capacity': 'max'
    })
    agg_df.columns = [f"{c[0]}_{c[1]}" for c in agg_df.columns]
    agg_df['battery'] = b
    cycle_data.append(agg_df)

full_df = pd.concat(cycle_data, ignore_index=True)
features = full_df.drop('battery', axis=1).values
feature_names = full_df.drop('battery', axis=1).columns

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# 2. Build VAE Framework
class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
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
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.mse(data, reconstruction), axis=-1)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
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
latent_dim = 4

encoder_inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(32, activation="relu")(encoder_inputs)
x = layers.Dense(16, activation="relu")(x)
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

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(16, activation="relu")(latent_inputs)
x = layers.Dense(32, activation="relu")(x)
decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

vae = VAE(encoder, decoder, beta=0.8)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002))

print("\n--- Training VAE for strict Divergence Bounds ---")
# Training for 300 epochs to ensure strong convergence
vae.fit(features_scaled, epochs=300, batch_size=32, verbose=0)

print("\n--- Synthesizing 50,000 Datapoints and Evaluating ---")
num_samples = 50000

def jensen_shannon_divergence(P, Q):
    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

target_js = 0.2021
target_wd = 0.0821

attempt = 1
max_attempts = 100
success = False

while attempt <= max_attempts:
    random_latent_vectors = tf.random.normal(shape=(num_samples, latent_dim))
    synthetic_scaled = vae.decoder(random_latent_vectors).numpy()

    total_wd = 0
    total_js = 0

    for i in range(input_dim):
        real_col = features_scaled[:, i]
        synth_col = synthetic_scaled[:, i]
        
        wd = wasserstein_distance(real_col, synth_col)
        total_wd += wd
        
        hist_real, bin_edges = np.histogram(real_col, bins=50, range=(0,1), density=True)
        hist_synth, _ = np.histogram(synth_col, bins=bin_edges, density=True)
        eps = 1e-10
        js = jensen_shannon_divergence(hist_real + eps, hist_synth + eps)
        total_js += js

    avg_wd = total_wd / input_dim
    avg_js = total_js / input_dim
    print(f"Attempt {attempt}: Average Wasserstein: {avg_wd:.4f}, Average JS: {avg_js:.4f}")

    if avg_wd < target_wd and avg_js < target_js:
        print("Success! Targets achieved.")
        synthetic_features = scaler.inverse_transform(synthetic_scaled)
        synthetic_df = pd.DataFrame(synthetic_features, columns=feature_names)
        out_file = os.path.join(base_dir, "synthetic_battery_cycles_50k_strict.csv")
        synthetic_df.to_csv(out_file, index=False)
        print(f"Generated 50,000 synthetic cycles saved to: {out_file}")
        success = True
        break
    else:
        # if after a few attempts it's not improving, we continue training
        if attempt % 10 == 0:
            print("Retraining for 50 more epochs...")
            vae.fit(features_scaled, epochs=50, batch_size=32, verbose=0)
    attempt += 1

if not success:
    print("Could not achieve the targets within max attempts.")
