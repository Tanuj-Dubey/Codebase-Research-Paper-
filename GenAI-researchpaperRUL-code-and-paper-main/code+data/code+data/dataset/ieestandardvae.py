import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance, entropy

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# Dataset Loading
# -------------------------
base_dir = r"d:\research paper - Copy\code+data\code+data\dataset\csv_format"
batteries = ["B0005","B0006","B0007","B0018"]

cycle_data = []

for b in batteries:

    df = pd.read_csv(os.path.join(base_dir,f"{b}_discharge.csv"))

    agg = df.groupby("id_cycle").agg({
        "Voltage_measured":["max","min","mean"],
        "Current_measured":["max","min","mean"],
        "Temperature_measured":"mean",
        "Capacity":"max"
    })

    agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns]
    agg["battery"] = b

    cycle_data.append(agg)

full_df = pd.concat(cycle_data,ignore_index=True)

features = full_df.drop("battery",axis=1).values.astype("float32")
feature_names = full_df.drop("battery",axis=1).columns

# -------------------------
# Normalization
# -------------------------
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

X_train,X_test = train_test_split(
    features_scaled,
    test_size=0.2,
    random_state=42
)

input_dim = X_train.shape[1]

# -------------------------
# VAE Architecture
# -------------------------
latent_dim = 8

# Encoder
encoder_inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(64,activation="relu")(encoder_inputs)
x = layers.Dense(32,activation="relu")(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):

    z_mean,z_log_var = args

    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0],latent_dim))

    return z_mean + tf.exp(0.5*z_log_var)*epsilon

z = layers.Lambda(sampling)([z_mean,z_log_var])

encoder = keras.Model(encoder_inputs,[z_mean,z_log_var,z])

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(32,activation="relu")(latent_inputs)
x = layers.Dense(64,activation="relu")(x)

decoder_outputs = layers.Dense(input_dim,activation="sigmoid")(x)

decoder = keras.Model(latent_inputs,decoder_outputs)

# -------------------------
# Custom VAE
# -------------------------
class VAE(keras.Model):

    def __init__(self,encoder,decoder,beta=0.8):

        super(VAE,self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def train_step(self,data):

        with tf.GradientTape() as tape:

            z_mean,z_log_var,z = self.encoder(data)

            reconstruction = self.decoder(z)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data-reconstruction),axis=1)
            )

            kl_loss = -0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss,axis=1))

            total_loss = recon_loss + self.beta*kl_loss

        grads = tape.gradient(total_loss,self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))

        return {"loss":total_loss}

vae = VAE(encoder,decoder)

vae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
)

# -------------------------
# Train Model
# -------------------------
print("\nTraining VAE...")

vae.fit(
    X_train,
    epochs=500,
    batch_size=32,
    verbose=1
)

# -------------------------
# Generate Synthetic Data
# -------------------------
num_samples = 50000

latent_vectors = tf.random.normal(shape=(num_samples,latent_dim))

synthetic_scaled = vae.decoder(latent_vectors).numpy()

synthetic = scaler.inverse_transform(synthetic_scaled)

synthetic_df = pd.DataFrame(
    synthetic,
    columns=feature_names
)

out_file = os.path.join(
    base_dir,
    "synthetic_battery_cycles_50k_ieee.csv"
)

synthetic_df.to_csv(out_file,index=False)

print("\nSynthetic dataset saved:",out_file)

# -------------------------
# Evaluation
# -------------------------
def js_divergence(P,Q):

    P = P/np.sum(P)
    Q = Q/np.sum(Q)

    M = 0.5*(P+Q)

    return 0.5*(entropy(P,M)+entropy(Q,M))

total_wd = 0
total_js = 0

print("\n--- Statistical Divergence Metrics ---\n")

for i,f in enumerate(feature_names):

    real = X_test[:,i]
    synth = synthetic_scaled[:,i]

    wd = wasserstein_distance(real,synth)

    hist_real,bins = np.histogram(real,bins=50,range=(0,1),density=True)
    hist_syn,_ = np.histogram(synth,bins=bins,density=True)

    js = js_divergence(hist_real+1e-10,hist_syn+1e-10)

    print(f"{f:30s} | WD={wd:.4f} | JS={js:.4f}")

    total_wd += wd
    total_js += js

avg_wd = total_wd/len(feature_names)
avg_js = total_js/len(feature_names)

print("\nAverage Wasserstein Distance:",round(avg_wd,4))
print("Average JS Divergence:",round(avg_js,4))

if avg_wd < 0.1 and avg_js < 0.25:
    print("\nAssessment: High morphological similarity.")
else:
    print("\nAssessment: Moderate similarity.")


