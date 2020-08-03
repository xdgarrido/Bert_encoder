"""BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import numpy as np
import os
# enable just error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import modeling

tf.compat.v1.disable_v2_behavior()

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("iter", 5000, "Total number of iterations")
flags.DEFINE_integer("prof_iter", 10, "Total number of iterations to be captured by profiler")
flags.DEFINE_integer("batch", 6, "Batch size")
flags.DEFINE_integer("seq_length", 512, "Sequence Length")
flags.DEFINE_integer("heads",16,"Number of heads")
flags.DEFINE_integer("layers",24,"Number of layers")
flags.DEFINE_string("mode","benchmark","Mode")
flags.DEFINE_string("precision","fp32","precision fp32 or fp16")

# batch and seq size that fit into a single GPU collected from https://github.com/ROCmSoftwarePlatform/BERT#out-of-memory-issues
batch_size = FLAGS.batch
seq_length = FLAGS.seq_length
heads = FLAGS.heads
layers = FLAGS.layers

if FLAGS.precision == "fp32":
# this is set to LARGE Bert model 
   bert_config = modeling.BertConfig(attention_probs_dropout_prob= 0.1,
      hidden_act= "gelu",
      hidden_dropout_prob= 0.1,
      hidden_size = 1024,
      initializer_range = 0.02,
      intermediate_size = 4096,
      max_position_embeddings = 512,
      num_attention_heads = heads,
      num_hidden_layers = layers,
      type_vocab_size =  2,
      vocab_size = 30522,
      precision=tf.float32)
else:
   bert_config = modeling.BertConfig(attention_probs_dropout_prob= 0.1,
      hidden_act= "gelu",
      hidden_dropout_prob= 0.1,
      hidden_size = 1024,
      initializer_range = 0.02,
      intermediate_size = 4096,
      max_position_embeddings = 512,
      num_attention_heads = heads,
      num_hidden_layers = layers,
      type_vocab_size =  2,
      vocab_size = 30522,
      precision=tf.float16)
# Create a TensorBoard log file
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Set the bert model input
input_ids   = tf.ones(shape=(batch_size, seq_length), dtype=tf.int32)
input_mask  = tf.ones(shape=(batch_size, seq_length), dtype=tf.int32)
token_ids   = tf.ones(shape=(batch_size, seq_length), dtype=tf.int32)

# Define to define loss
hidden_size = 1024
labels = tf.ones(shape=(batch_size,), dtype=tf.int32)

bert_model = modeling.BertModel(
      config=bert_config,
      is_training=True,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=token_ids,
      use_one_hot_embeddings=False)

# Finalize bert
output_layer = bert_model.get_pooled_output()
logits = tf.compat.v1.layers.dense(output_layer, units=hidden_size, activation=tf.nn.softmax)
# This is just to compute backward pass
loss   = tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
tvars  = tf.compat.v1.trainable_variables()
global_step = tf.compat.v1.train.get_or_create_global_step()
# Any optimizer will do it, picking a super light one.
opt          = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=2e-5)
grads_and_vars       = opt.compute_gradients(loss,tvars)


encoder_train = opt.apply_gradients(grads_and_vars, global_step=global_step)


init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                   tf.compat.v1.local_variables_initializer())

# fire-up bert
i = 0
with tf.compat.v1.Session() as sess:
  sess.run(init_op)
  with tf.device('/GPU:0'):
    while i < (FLAGS.iter-FLAGS.prof_iter):
      sess.run(encoder_train) 
      i = i+1
    tf.profiler.experimental.start(logs)
    while i < (FLAGS.iter):
      sess.run(encoder_train) 
      i = i+1
tf.profiler.experimental.stop()
print("Final iteration is ", i)