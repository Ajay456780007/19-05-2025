import math
import time
import matplotlib.pyplot as plt
import gzip
import seaborn as sns
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from urllib.parse import unquote
import numpy as np
from keras import Sequential, layers, Model
from collections import Counter
from keras.src.layers import Bidirectional, LSTM, Dense, Dropout
from keras.src.metrics.accuracy_metrics import accuracy
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras import layers
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical
from keras import mixed_precision
import os

import torch
from torch import nn
from layer_HGNN import HGNN_conv
import torch.nn.functional as F



# === Paths ===
def read_data():
    import pandas as pd
    import numpy as np
    import os
    import gzip
    from urllib.parse import unquote
    from Bio import SeqIO
    from Bio.Seq import Seq
    from sklearn.preprocessing import StandardScaler

    dna_dir = "dataset/dataset1/dna_chromosomes/"
    gff3_dir = "dataset/dataset1/gff3_files/"

    # === Collect all FASTA and GFF3 files ===
    fasta_files = sorted([
        os.path.join(dna_dir, f) for f in os.listdir(dna_dir)
        if f.lower().endswith(".fa.gz")
    ])
    gff3_files = sorted([
        os.path.join(gff3_dir, f) for f in os.listdir(gff3_dir)
        if f.lower().endswith(".gff3")
    ])

    # === Parse GFF3 attributes ===
    def parse_attributes(attr_str):
        attr_dict = {}
        for pair in attr_str.strip().split(";"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                attr_dict[key.strip()] = unquote(value.strip())
        return attr_dict

    # === Function to parse GFF3 and extract CDS entries for a given chromosome ===
    def parse_gff3_cds(gff3_file, chrom_id):
        cds_dict = {}
        with open(gff3_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                if parts[2] != "CDS":
                    continue
                if parts[0] != chrom_id:
                    continue

                start = int(parts[3]) - 1  # GFF3 is 1-based; convert to 0-based
                end = int(parts[4])  # end is exclusive
                strand = parts[6]
                attrs = parse_attributes(parts[8])
                parent_id = attrs.get("Parent", "NA")

                if parent_id not in cds_dict:
                    cds_dict[parent_id] = {
                        "strand": strand,
                        "ranges": []
                    }
                cds_dict[parent_id]["ranges"].append((start, end))
        return cds_dict

    # === Extract CDS sequences with start/end positions ===
    cds_sequences = []

    for fasta_path, gff_path in zip(fasta_files, gff3_files):
        print(f"Processing: {os.path.basename(fasta_path)} with {os.path.basename(gff_path)}")

        # Read chromosome sequence
        with gzip.open(fasta_path, "rt", encoding="utf-8") if fasta_path.endswith(".gz") else open(fasta_path, "r",
                                                                                                   encoding="utf-8") as f:
            record = next(SeqIO.parse(f, "fasta"))

        chrom_seq = record.seq
        chrom_id = record.id

        # Parse CDS entries
        cds_dict = parse_gff3_cds(gff_path, chrom_id)
        print(f"Found {len(cds_dict)} CDS transcripts in {os.path.basename(gff_path)}")

        for parent_id, info in cds_dict.items():
            strand = info["strand"]
            regions = sorted(info["ranges"], key=lambda x: x[0])
            full_seq = "".join(str(chrom_seq[start:end]) for start, end in regions)
            if strand == "-":
                full_seq = str(Seq(full_seq).reverse_complement())

            transcript_start = min(start for start, end in regions)
            transcript_end = max(end for start, end in regions)

            cds_sequences.append({
                "transcript_id": parent_id,
                "chrom": chrom_id,
                "strand": strand,
                "start": transcript_start,
                "end": transcript_end,
                "sequence": full_seq
            })

    df_cds = pd.DataFrame(cds_sequences)
    df_cds.to_csv("dataset/dataset1/cds_sequences.csv", index=False)

    # === Load mapping and expression data ===
    alias_df = pd.read_csv("dataset/dataset1/geo_files/genes_to_alias_ids.tsv", sep="\t", header=None)
    tpm_df = pd.read_csv("dataset/dataset1/expression level TPM/abundance.tsv", sep="\t")

    alias_df.columns = ["e_id", "source", "d_id", "agpv4_id"]
    tpm_df["gene_id"] = tpm_df["target_id"].apply(lambda x: x.split("_T")[0])
    df_cds["clean_gene_id"] = df_cds["transcript_id"].apply(lambda x: x.replace("transcript:", "").split("_T")[0])

    df_merged = df_cds.merge(alias_df[["e_id", "d_id"]], left_on="clean_gene_id", right_on="e_id", how="left")

    avg_tpm = tpm_df.groupby("gene_id", as_index=False)["tpm"].mean().rename(columns={"tpm": "avg_tpm"})
    final_df = df_merged.merge(avg_tpm, left_on="d_id", right_on="gene_id", how="left")

    final_df = final_df.drop(columns=["clean_gene_id", "e_id", "gene_id"])
    final_df.dropna(subset=['avg_tpm'], inplace=True)

    # === Normalize start and end columns using z-score
    scaler = StandardScaler()
    final_df[["start_z", "end_z"]] = scaler.fit_transform(final_df[["start", "end"]])

    # === Encode transcript_id, strand, sequence
    final_df["transcript_index"] = pd.factorize(final_df["transcript_id"])[0]
    final_df["strand_numeric"] = final_df["strand"].map({'+': 1, '-': 0})

    def encode_sequence(seq):
        mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        return [mapping.get(base.upper(), 4) for base in seq]

    final_df["sequence_encoded"] = final_df["sequence"].apply(encode_sequence)

    # === Convert avg_tpm to expression labels
    def compute_labels(tpm_array):
        mean_tpm = np.mean(tpm_array)
        low = mean_tpm / 2
        high = mean_tpm * 1.5
        return np.array([0 if t < low else 1 if t < high else 2 for t in tpm_array], dtype=np.int32)

    final_df["expression_label"] = compute_labels(final_df["avg_tpm"].values)

    # === Pad or truncate sequences
    FIXED_LEN = 6000
    PAD_VALUE = 4

    def pad_or_truncate(seq):
        return seq[:FIXED_LEN] if len(seq) > FIXED_LEN else seq + [PAD_VALUE] * (FIXED_LEN - len(seq))

    final_df['sequence_encoded'] = final_df['sequence_encoded'].apply(pad_or_truncate)

    # === Final feature selection and saving
    # Save X and y
    np.save('dataset/dataset1/sequence_encoded.npy', np.array(final_df['sequence_encoded'].tolist(), dtype=np.int32))
    np.save('dataset/dataset1/expression_label.npy', final_df['expression_label'].values.astype(np.int32))

    # Save other features (excluding sequence, label, and transcript index)
    other_features = final_df.drop(columns=['sequence_encoded', 'expression_label', 'transcript_index','transcript_id',"sequence","avg_tpm","start","end","strand","d_id"])
    np.save('dataset/dataset1/other_features.npy', other_features.values)

    # === Logging info ===
    print("Unique label values:", final_df["expression_label"].unique())
    print("Transcript ID sample:", final_df["transcript_id"].head())
    print("other_features shape:", other_features.shape)
    print("sequence_encoded shape:", np.load('dataset/dataset1/sequence_encoded.npy').shape)
    print("expression_label shape:", np.load('dataset/dataset1/expression_label.npy').shape)


#
# read_data()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# === Load Only First 50 Samples Directly ===
sequence_data = np.load('dataset/dataset1/sequence_encoded.npy')[:2000, :6000]
expression_labels = np.load("dataset/dataset1/expression_label.npy")[:2000].astype(int)
node_features = np.load('dataset/dataset1/other_features.npy',allow_pickle=True)[:20000]  # shape (50, 5)

node_features = np.array(list(node_features)).astype(np.float32)
# # === Create Shared Hypergraph Matrix ===
# def create_random_hypergraph(num_nodes, num_hyperedges, connection_prob=0.1):
#     return np.random.rand(num_nodes, num_hyperedges) < connection_prob
#
# shared_G = create_random_hypergraph(50, 50).astype(np.float32)  # (50, 50)
# shared_G_batch = tf.convert_to_tensor(shared_G[np.newaxis, ...])  # (1, 50, 50)
#
# # === Train/Test Split (same for transformer and HGNN) ===
# indices = np.arange(50)
# train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=expression_labels, random_state=42)
#
# x_train_dna = sequence_data[train_idx]
# x_test_dna = sequence_data[test_idx]
# x_train_node = node_features[train_idx]
# x_test_node = node_features[test_idx]
# y_train = expression_labels[train_idx]
# y_test = expression_labels[test_idx]
#
# from sklearn.model_selection import train_test_split
#
# # Split 20% of the training data for validation
# x_train_dna, x_val_dna, x_train_node, x_val_node, y_train, y_val = train_test_split(
#     x_train_dna,
#     x_train_node,
#     y_train,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_train  # Keep class distribution balanced
# )
x=sequence_data #shape(100,6000)
y=expression_labels #shape(100)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# === Positional Encoding and Transformer Components ===
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, Model, Input
# from tensorflow.keras.layers import Dense, Dropout

# --- Positional Encoding ---
# def get_positional_encoding(seq_len, model_dim):
#     angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(10000, (
#         2 * (np.arange(model_dim)[np.newaxis, :] // 2)) / model_dim)
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#     return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)
#
# # --- Sinusoidal Time Embedding ---
# class SinusoidalEmbedding(layers.Layer):
#     def __init__(self, model_dim):
#         super().__init__()
#         self.model_dim = model_dim
#
#     def call(self, x):
#         half_dim = self.model_dim // 2
#         freqs = tf.exp(tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), half_dim))
#         angles = 2.0 * np.pi * x * freqs
#         return tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)[..., tf.newaxis]
#
# # --- Transformer Block ---
# class TransformerBlock(layers.Layer):
#     def __init__(self, model_dim, heads, ff_dim, rate=0.1):
#         super().__init__()
#         self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=model_dim // heads, dropout=rate)
#         self.ffn = tf.keras.Sequential([
#             layers.Dense(ff_dim, activation='gelu'),
#             layers.Dense(model_dim),
#         ])
#         self.norm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.norm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(rate)
#         self.dropout2 = layers.Dropout(rate)
#
#     def call(self, x, training=False):
#         x = self.norm1(x + self.dropout1(self.att(x, x), training=training))
#         return self.norm2(x + self.dropout2(self.ffn(x), training=training))
#
# # --- Diffusion Transformer Core ---
# class DiffusionTransformer(tf.keras.Model):
#     def __init__(self, seq_len=6000, model_dim=128, num_heads=4, ff_dim=256, depth=4, dropout_rate=0.1):
#         super().__init__()
#         self.embedding = layers.Embedding(input_dim=5, output_dim=model_dim)  # DNA ACGTN → 0-4
#         self.pos_encoding = get_positional_encoding(seq_len, model_dim)
#         self.time_emb = SinusoidalEmbedding(model_dim)
#         self.transformer_blocks = [TransformerBlock(model_dim, num_heads, ff_dim, dropout_rate) for _ in range(depth)]
#         self.global_pool = layers.GlobalAveragePooling1D()
#
#     def call(self, x, training=False):
#         noise_var = tf.zeros((tf.shape(x)[0], 1))
#         x = self.embedding(x)
#         x += self.pos_encoding[:, :tf.shape(x)[1], :]
#         noise_emb = self.time_emb(noise_var)
#         noise_emb = tf.transpose(noise_emb, [0, 2, 1])
#         noise_emb = tf.tile(noise_emb, [1, tf.shape(x)[1], 1])
#         x += tf.cast(noise_emb, tf.float32)
#         for block in self.transformer_blocks:
#             x = block(x, training=training)
#         return self.global_pool(x)
#
# # --- Build Full Model for Classification ---
# def build_diffusion_model(seq_len=6000, model_dim=128, num_classes=3):
#     inputs = Input(shape=(seq_len,))
#     core_model = DiffusionTransformer(seq_len=seq_len, model_dim=model_dim)
#     x = core_model(inputs)
#     x = Dropout(0.3)(x)
#     outputs = Dense(num_classes, activation='softmax')(x)
#     return Model(inputs=inputs, outputs=outputs)
# y_train = np.squeeze(y_train)
# y_test = np.squeeze(y_test)
# print(x_train.shape)  # (80, 6000)
# print(y_train.shape)  # (80,)
# print(np.unique(y_train))
# model = build_diffusion_model()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5, batch_size=8)


# === HGNN Components ===
# class HGNNConv(layers.Layer):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.linear = layers.Dense(output_dim, use_bias=False)
#
#     def call(self, x, G):
#         x = self.linear(x)
#         return tf.matmul(tf.transpose(G, [0, 2, 1]), x)
#
# class HGNNEmbedding(tf.keras.Model):
#     def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.5):
#         super().__init__()
#         self.hgc1 = HGNNConv(input_dim, hidden_dim)
#         self.hgc2 = HGNNConv(hidden_dim, hidden_dim)
#         self.dropout = layers.Dropout(dropout_rate)
#
#     def call(self, x, G, training=False):
#         x = self.hgc1(x, G)
#         x = self.dropout(x, training=training)
#         x = self.hgc2(x, G)
#         return tf.reduce_mean(x, axis=1)
#
# # === Fusion and Combined Model ===
# class SimpleFusionModel(tf.keras.Model):
#     def __init__(self, fusion_dim=128, num_classes=3, dropout_rate=0.2):
#         super().__init__()
#         self.dense1 = layers.Dense(fusion_dim, activation='relu')
#         self.dropout = layers.Dropout(dropout_rate)
#         self.output_layer = layers.Dense(num_classes, activation='softmax')
#
#     def call(self, x, training=False):
#         x = self.dense1(x)
#         x = self.dropout(x, training=training)
#         return self.output_layer(x)
#
# class CombinedModel(tf.keras.Model):
#     def __init__(self, diffusion_transformer, hgnn_embedding, fusion_model):
#         super().__init__()
#         self.diffusion_transformer = diffusion_transformer
#         self.hgnn_embedding = hgnn_embedding
#         self.fusion_model = fusion_model
#
#     def call(self, inputs, training=False):
#         dna_seq, node_feat = inputs
#         batch_size = tf.shape(dna_seq)[0]
#         G = tf.tile(shared_G_batch, [batch_size, 1, 1])
#         node_feat_tiled = tf.tile(tf.expand_dims(node_feat, 1), [1, 50, 1])
#         dt_out = self.diffusion_transformer(dna_seq, training=training)
#         hg_out = self.hgnn_embedding(node_feat_tiled, G, training=training)
#         combined = tf.concat([dt_out, hg_out], axis=-1)
#         return self.fusion_model(combined, training=training)
#
# # === Build, Compile, Train, Evaluate ===
# diffusion_transformer = DiffusionTransformer(seq_len=6000)
# hgnn_embedding = HGNNEmbedding(input_dim=node_features.shape[1])
# fusion_model = SimpleFusionModel(fusion_dim=128, num_classes=len(np.unique(expression_labels)))
# from sklearn.utils.class_weight import compute_class_weight
#
# # --- Compute class weights ---
# class_labels = np.unique(y_train)
# class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
# class_weight_dict = dict(zip(class_labels, class_weights))
#
# model = CombinedModel(diffusion_transformer, hgnn_embedding, fusion_model)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # === Train ===
# model.fit(
#     (x_train_dna, x_train_node),
#     y_train,
#     batch_size=2,
#     epochs=10,
#     validation_data=((x_val_dna, x_val_node), y_val),
#     class_weight=class_weight_dict
# )

# === Evaluate ===
# loss, acc = model.evaluate((x_test_dna, x_test_node), y_test)
# print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# import numpy as np
# from collections import Counter
# from sklearn.metrics import accuracy_score
#
# class KNN:
#     def __init__(self, k):
#         self.k = k
#         print(f"KNN initialized with k = {self.k}")
#
#     def fit(self, X_train, y_train):
#         if self.k > len(X_train):
#             raise ValueError("k cannot be greater than the number of training samples")
#         self.x_train = np.array(X_train)
#         self.y_train = np.array(y_train).flatten()
#
#     def calculate_euclidean(self, sample1, sample2):
#         return np.linalg.norm(sample1.astype(np.float32) - sample2.astype(np.float32))
#
#     def nearest_neighbors(self, test_sample):
#         distances = [
#             (self.y_train[i], self.calculate_euclidean(self.x_train[i], test_sample))
#             for i in range(len(self.x_train))
#         ]
#         distances.sort(key=lambda x: x[1])  # Sort by distance
#         neighbors = [distances[i][0] for i in range(self.k)]
#         return neighbors
#
#     def majority_vote(self, neighbors):
#         count = Counter(neighbors)
#         return sorted(count.items(), key=lambda x: (-x[1], x[0]))[0][0]
#
#     def predict(self, test_set):
#         predictions = []
#         for test_sample in test_set:
#             neighbors = self.nearest_neighbors(test_sample)
#             prediction = self.majority_vote(neighbors)
#             predictions.append(prediction)
#         return predictions
# #KNN Model Building
# # === Apply KNN to your dataset ===
# # Make sure x_train, x_test, y_train, y_test are already defined
# #
# # model3 = KNN(k=5)
# # model3.fit(x_train, y_train)
# # predictions = model3.predict(x_test)
# #
# # accuracy = accuracy_score(y_test, predictions)
# # print(f"Accuracy for KNN: {accuracy:.4f}")
#
#
#
#
#
# def create_BiLSTM(input_shape, num_classes):
#     model = Sequential()
#     model.add(Bidirectional(LSTM(units=64,
#                                  return_sequences=False,
#                                  activation='tanh'),
#                             input_shape=input_shape))
#     model.add(Dense(units=num_classes, activation='softmax'))
#
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                   metrics=["accuracy"])
#     return model
#
# #BiLSTM Model Buildimg
#
# # # Number of classes in your classification
# # num_classes = 3  # e.g., low = 0, medium = 1, high = 2
#
#
# # model2 = create_BiLSTM(x_train_bilstm.shape[1:], num_classes)
# # model2.fit(x_train_bilstm, y_train, epochs=10, batch_size=32, validation_split=0.1)
# #
# #
# # loss_bilstm, acc_bilstm = model2.evaluate(x_test_bilstm, y_test)
# # print("The loss of BiLSTM:", loss_bilstm)
# # print("The accuracy of BiLSTM:", acc_bilstm)
#
#
#
# def compute_metrics(y_true, y_pred, average='macro'):
#     cm = confusion_matrix(y_true, y_pred)
#     tp = np.diag(cm)
#     fn = cm.sum(axis=1) - tp
#     fp = cm.sum(axis=0) - tp
#     tn = cm.sum() - (tp + fn + fp)
#     specificity = np.mean(tn / (tn + fp)) if np.all(tn + fp) else 0.0
#
#     return {
#         "confusion_matrix": cm,
#         "accuracy": accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
#         "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
#         "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
#         "specificity": specificity,
#         "mae": mean_absolute_error(y_true, y_pred),
#         "mse": mean_squared_error(y_true, y_pred)
#     }
#
# # === Main Evaluation Loop ===
# results = {"ProposedModel": [], "KNN": [], "BiLSTM": []}
# metrics = {"ProposedModel": [], "KNN": [], "BiLSTM": []}
# training_percentage = [40, 50, 60, 70, 80, 90]
#
# #
# x_all = np.concatenate([x_train, x_test], axis=0)
# y_all = np.concatenate([y_train, y_test], axis=0)
#
# for percent in training_percentage:
#     print(f"\n=== Training with {percent}% of total data ===")
#     indices = np.arange(len(x_all))
#     np.random.shuffle(indices)
#     num_train = int(len(x_all) * percent / 100)
#
#
#     train_idx = indices[:num_train]
#     test_idx = indices[num_train:]
#
#
#     def get_generators(x_all, y_all, node_features_reduced, hg_adj, train_idx, test_idx, batch_size=32):
#         x_train = x_all[train_idx]
#         y_train = y_all[train_idx]
#         x_test = x_all[test_idx]
#         y_test = y_all[test_idx]
#
#         node_features_train = node_features_reduced[train_idx]
#         node_features_test = node_features_reduced[test_idx]
#         hg_adj_train = hg_adj[train_idx]
#         hg_adj_test = hg_adj[test_idx]
#
#         # Create data generators
#         train_gen = HybridDataGenerator(x_train, node_features_train, hg_adj_train, y_train,
#                                         batch_size=batch_size, shuffle=True)
#         test_gen = HybridDataGenerator(x_test, node_features_test, hg_adj_test, y_test,
#                                        batch_size=batch_size, shuffle=False)
#
#         return train_gen, test_gen, y_test
#
#
#     print(
#         f"x_train: {x_train.shape}, node_features_train: {node_features_train.shape}, hg_adj_train: {hg_adj_train.shape}")
#     print(f"x_test: {x_test.shape}, node_features_test: {node_features_test.shape}, hg_adj_test: {hg_adj_test.shape}")
#
#     # --- Proposed Model ---
#     train_gen, test_gen, y_test_split = get_generators(x_all, y_all, node_features_reduced, hg_adj, train_idx, test_idx,
#                                                        batch_size=32)
#
#     # --- Proposed Model ---
#     combined_model = CombinedModel(diffusion_transformer, hgnn_embedding, fusion_dim=64, num_classes=3)
#     combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                            loss='sparse_categorical_crossentropy',
#                            metrics=['accuracy'])
#
#     combined_model.fit(train_gen, validation_data=test_gen, epochs=10, verbose=1)
#
#     # Predict on test set using generator
#     y_pred_probs = combined_model.predict(test_gen)
#     y_pred = np.argmax(y_pred_probs, axis=1)
#     metric_vals = compute_metrics(y_test, y_pred)
#     results["ProposedModel"].append(metric_vals["accuracy"])
#     metrics["ProposedModel"].append(metric_vals)
#     print(f"ProposedModel Accuracy: {metric_vals['accuracy']:.4f}")
#
#     # #--- BiLSTM Model---
#     # x_train_bilstm = np.expand_dims(x_train, axis=-1).astype(np.float32)
#     # x_test_bilstm = np.expand_dims(x_test, axis=-1).astype(np.float32)
#     # model2 = create_BiLSTM(x_train_bilstm.shape[1:], num_classes=3)
#     # model2.fit(x_train_bilstm, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
#     # y_pred = np.argmax(model2.predict(x_test_bilstm), axis=-1)
#     # metric_vals = compute_metrics(y_test, y_pred)
#     # results["BiLSTM"].append(metric_vals["accuracy"])
#     # metrics["BiLSTM"].append(metric_vals)
#     # print(f"BiLSTM Accuracy: {metric_vals['accuracy']:.4f}")
#     #
#     # # --- KNN ---
#     # knn_model = KNN(k=5)
#     # knn_model.fit(x_train, y_train)
#     # y_pred = knn_model.predict(x_test)
#     # metric_vals = compute_metrics(y_test, y_pred)
#     # results["KNN"].append(metric_vals["accuracy"])
#     # metrics["KNN"].append(metric_vals)
#     # print(f"KNN Accuracy: {metric_vals['accuracy']:.4f}")
#
# # === Save results ===
# np.save("model_accuracy_results.npy", results)
# np.save("model_detailed_metrics.npy", metrics)
#
# # === Accuracy Plot ===
# bar_width = 0.2
# x_range = np.arange(len(training_percentage))
# model_names = list(results.keys())
# plt.figure(figsize=(12, 6))
#
# for i, model_name in enumerate(model_names):
#     plt.bar(x_range + i * bar_width, results[model_name], width=bar_width, label=model_name)
#
# plt.xlabel("Training Percentage")
# plt.ylabel("Accuracy")
# plt.title("Model Accuracy vs Training Data Percentage")
# plt.xticks(x_range + bar_width, training_percentage)
# plt.legend()
# plt.tight_layout()
# plt.savefig("training_percentage_comparison_bar.png")
# plt.show()
#
# # === Confusion Matrices ===
# for model_name in model_names:
#     for i, percent in enumerate(training_percentage):
#         cm = metrics[model_name][i]["confusion_matrix"]
#         plt.figure(figsize=(6, 5))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.title(f"{model_name} Confusion Matrix ({percent}%)")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.tight_layout()
#         plt.savefig(f"conf_matrix_{model_name}_{percent}.png")
#         plt.close()


x = torch.tensor(node_features, dtype=torch.float32)

y = torch.tensor(y, dtype=torch.long)
import numpy as np


# def build_incidence_matrix_from_labels(y):
#     """
#     Build incidence matrix H from TPM class labels.
#     Each unique label defines a hyperedge.
#
#     Parameters:
#         y (Tensor): Tensor of shape [num_genes], with labels like 0,1,2
#
#     Returns:
#         H (ndarray): [num_genes, num_hyperedges]
#     """
#     y_np = y.cpu().numpy() if hasattr(y, 'cpu') else y  # Convert to numpy if tensor
#     num_genes = len(y_np)
#     classes = np.unique(y_np)
#     num_hyperedges = len(classes)
#
#     H = np.zeros((num_genes, num_hyperedges), dtype=np.float32)
#
#     for j, label in enumerate(classes):
#         H[:, j] = (y_np == label).astype(float)  # mark genes in this class
#     return H
#
#
# def generate_G_from_H(H):
#     H = torch.tensor(H)
#     Dv = torch.diag(torch.sum(H, dim=1))  # Vertex degrees
#     De = torch.diag(torch.sum(H, dim=0))  # Hyperedge degrees
#     De_inv = torch.inverse(De)
#     Dv_inv_sqrt = torch.inverse(torch.sqrt(Dv))
#     HT = torch.transpose(H, 0, 1)
#     G = Dv_inv_sqrt @ H @ De_inv @ HT @ Dv_inv_sqrt
#     return G
#
#
# class HGNN(nn.Module):
#     def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
#         super(HGNN, self).__init__()
#         self.dropout = dropout
#         self.hgc1 = HGNN_conv(in_ch, n_hid)
#         self.hgc2 = HGNN_conv(n_hid, n_class)
#
#     def forward(self, x, G):
#         x = F.relu(self.hgc1(x, G))
#         x = F.dropout(x, self.dropout)
#         x = self.hgc2(x, G)
#         return x
#
#
# H = build_incidence_matrix_from_labels(y)
# G = generate_G_from_H(H)
#
#
#
# in_ch=x.shape[1]
# n_class=len(torch.unique(y))
# idx = torch.arange(len(y))
# n_hid=128
#
# model=HGNN(in_ch,n_class,n_hid,dropout=0.5)
# optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
# criterion=nn.CrossEntropyLoss()
#
# from sklearn.model_selection import train_test_split
#
# idx = torch.arange(len(y))
# train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
#
# x_train, y_train = x[train_idx], y[train_idx]
# x_test, y_test = x[test_idx], y[test_idx]

# epochs = 50
# model.train()
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     out = model(x, G)  # Entire graph used, but only train on train_idx
#     loss = criterion(out[train_idx], y_train)
#     loss.backward()
#     optimizer.step()
#
#
#     print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
#
# model.eval()
# with torch.no_grad():
#     preds = model(x, G)
#     pred_classes = preds[test_idx].argmax(dim=1)
#     acc = (pred_classes == y_test).float().mean()
#     print(f"Test Accuracy: {acc:.4f}")
#


# diffusion transformer--

import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LinformerAttention(nn.Module):
    def __init__(self, seq_len, dim, n_heads, k, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qw = nn.Linear(dim, dim, bias=bias)
        self.kw = nn.Linear(dim, dim, bias=bias)
        self.vw = nn.Linear(dim, dim, bias=bias)
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        self.ow = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)
        B, L, D = q.shape
        q = q.view(B, L, self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.view(B, L, self.n_heads, -1).permute(0, 2, 3, 1)
        v = v.view(B, L, self.n_heads, -1).permute(0, 2, 3, 1)

        k = torch.matmul(k, self.E[:L, :])
        v = torch.matmul(v, self.F[:L, :]).permute(0, 1, 3, 2)

        qk = torch.matmul(q, k) * self.scale
        attn = torch.softmax(qk, dim=-1)

        v_attn = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, L, D)
        return self.ow(v_attn)


class TransformerBlock(nn.Module):
    def __init__(self, seq_len, dim, heads, mlp_dim, k, rate=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = LinformerAttention(seq_len, dim, heads, k)
        self.ln_2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate),
        )
        self.gamma_1 = nn.Linear(dim, dim)
        self.beta_1 = nn.Linear(dim, dim)
        self.gamma_2 = nn.Linear(dim, dim)
        self.beta_2 = nn.Linear(dim, dim)
        self.scale_1 = nn.Linear(dim, dim)
        self.scale_2 = nn.Linear(dim, dim)

        self._init_weights([self.gamma_1, self.beta_1, self.gamma_2,
                            self.beta_2, self.scale_1, self.scale_2])

    def _init_weights(self, layers):
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, c):
        scale_msa = self.gamma_1(c)
        shift_msa = self.beta_1(c)
        scale_mlp = self.gamma_2(c)
        shift_mlp = self.beta_2(c)
        gate_msa = self.scale_1(c).unsqueeze(1)
        gate_mlp = self.scale_2(c).unsqueeze(1)

        x = self.attn(modulate(self.ln_1(x), shift_msa, scale_msa)) * gate_msa + x
        x = self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)) * gate_mlp + x
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

        self._init_weights([self.linear, self.gamma, self.beta])

    def _init_weights(self, layers):
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, c):
        scale = self.gamma(c)
        shift = self.beta(c)
        x = modulate(self.ln_final(x), shift, scale)
        return self.linear(x)  # (B, L, out_dim)


class DiT1D(nn.Module):
    def __init__(self, seq_len, dim=128, depth=3, heads=4, mlp_dim=512, k=64, input_dim=4, output_dim=3):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim

        self.embedding = nn.Linear(input_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))

        self.emb = nn.Sequential(
            PositionalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.transformer = nn.ModuleList([
            TransformerBlock(seq_len, dim, heads, mlp_dim, k)
            for _ in range(depth)
        ])

        self.final = FinalLayer(dim, output_dim)

    def forward(self, x, t):
        """
        x: (B, L, input_dim) - DNA sequence input
        t: (B,) - time or condition token (e.g., expression context)
        """
        x = self.embedding(x) + self.pos_embedding  # (B, L, dim)
        t = self.emb(t)  # (B, dim)

        for block in self.transformer:
            x = block(x, t)

        out = self.final(x, t)  # (B, L, output_dim)
        return out



import torch


def get_scalings(sig, sig_data):
    s = sig ** 2 + sig_data ** 2
    c_skip = sig_data ** 2 / s
    c_out = sig * sig_data / s.sqrt()
    c_in = 1 / s.sqrt()
    return c_skip, c_out, c_in


def get_sigmas_karras(n, sigma_min=0.01, sigma_max=80., rho=7., device='cpu'):
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.tensor([0.], device=device)])


class Diffusion(object):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.66):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def diffuse(self, y):
        device = y.device
        # For DNA: y shape = [B, C, L]
        B = y.shape[0]
        sigma = torch.exp(torch.randn([B, 1, 1], device=device) * self.P_std + self.P_mean)
        n = torch.randn_like(y)
        c_skip, c_out, c_in = get_scalings(sigma, self.sigma_data)
        noised_input = y + n * sigma
        target = (y - c_skip * noised_input) / c_out
        return c_in * noised_input, sigma.view(-1), target

    def sample(self, model, sz, steps=100, sigma_max=80., seed=None):
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                return self._sample_internal(model, sz, steps, sigma_max)
        else:
            return self._sample_internal(model, sz, steps, sigma_max)

    def _sample_internal(self, model, sz, steps, sigma_max):
        device = next(model.parameters()).device
        model.eval()
        x = torch.randn(sz, device=device) * sigma_max
        t_steps = get_sigmas_karras(steps, device=device, sigma_max=sigma_max)

        for i in range(len(t_steps) - 1):
            x = self.edm_sampler(x, t_steps, i, model)
        return x.cpu()

    @torch.no_grad()
    def edm_sampler(self, x, t_steps, i, model, s_churn=0., s_min=0.,
                    s_max=float('inf'), s_noise=1.):
        n = len(t_steps)
        gamma = self.get_gamma(t_steps[i], s_churn, s_min, s_max, s_noise, n)
        eps = torch.randn_like(x) * s_noise
        t_hat = t_steps[i] + gamma * t_steps[i]
        if gamma > 0:
            x_hat = x + eps * (t_hat ** 2 - t_steps[i] ** 2).sqrt()
        else:
            x_hat = x
        d = self.get_d(model, x_hat, t_hat)
        d_cur = (x_hat - d) / t_hat
        x_next = x_hat + (t_steps[i + 1] - t_hat) * d_cur
        if t_steps[i + 1] != 0:
            d = self.get_d(model, x_next, t_steps[i + 1])
            d_prime = (x_next - d) / t_steps[i + 1]
            d_prime = (d_cur + d_prime) / 2
            x_next = x_hat + (t_steps[i + 1] - t_hat) * d_prime
        return x_next

    def get_d(self, model, x, sig):
        B = x.shape[0]
        sig = sig.view(B, 1, 1)  # shape broadcastable to [B, C, L]
        c_skip, c_out, c_in = get_scalings(sig, self.sigma_data)
        return model(x * c_in, sig.view(B)) * c_out + x * c_skip

    def get_gamma(self, t_cur, s_churn, s_min, s_max, s_noise, n):
        if s_min <= t_cur <= s_max:
            return min(s_churn / (n - 1), 2 ** 0.5 - 1)
        else:
            return 0.


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure x_copy is a tensor of type long
if isinstance(x_copy, torch.Tensor):
    x = x_copy.to(torch.long)
else:
    x = torch.tensor(x_copy, dtype=torch.long)

# Set the correct number of classes (usually 4 for A, C, G, T; 5 if including padding/unknown)
num_classes = 5  # Use 5 if values go from 0 to 4 (0:A, 1:C, 2:G, 3:T, 4:padding/unknown)

# One-hot encode the input
x_dna = F.one_hot(x, num_classes=num_classes).float()  # Shape: (B, L, C)

# Print shape and unique values for debugging
print("x_dna shape:", x_dna.shape)       # Expected: (20000, 6000, 5)
print("Unique values in x:", x.unique()) # Values in the original sequence
print("Unique values in x_dna:", x_dna.unique())  # Should be 0.0 and 1.0 only

 # Shape: (20000, 6000, 4)



print("x shape:", x_dna.shape)  # or x_dna.shape
print("Unique values in x",x_dna.unique())

from torch.utils.data import Dataset, DataLoader

class DNADataset(Dataset):
    def __init__(self, x_data):
        self.x_data = x_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        if x.ndim == 2:
            return x  # already (L, C)
        elif x.ndim == 1:
            return x.unsqueeze(-1)  # from (L,) → (L, 1)
        else:
            raise ValueError(f"Unexpected x shape: {x.shape}")


dataset = DNADataset(x_dna)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DiT1D(
    seq_len=x.shape[1],     # 6000
    dim=128,
    depth=3,
    heads=4,
    mlp_dim=512,
    k=64,
    input_dim=x.shape[2],   # 4 (one-hot channels)
    output_dim=3
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

diffusion = Diffusion()
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = batch.to(device)  # batch shape: [B, L, C]

        # Diffuse data: add noise and get target denoising signal
        x_noisy, sigmas, target = diffusion.diffuse(batch.permute(0, 2, 1))
        # Note: diffuse expects shape [B, C, L], so permute if needed
        x_noisy = x_noisy.permute(0, 2, 1)  # back to [B, L, C]
        target = target.permute(0, 2, 1)

        # Forward pass
        pred = model(x_noisy, sigmas)  # sigmas shape: (B,)

        # Loss (e.g. MSE between pred and target)
        loss = nn.MSELoss()(pred, target)

        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
