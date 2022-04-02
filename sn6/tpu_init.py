### INIT TPU - MUST DO BEFORE SECRETS FOR PRIVATE DATASET###
import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print('Device:', tpu.master())
    strategy = tf.distribute.experimental.TPUStrategy(tpu)  # set distribution strategy
except:
    tpu = None
    print('not using TPU')
    strategy = tf.distribute.get_strategy()
print("REPLICAS: ", strategy.num_replicas_in_sync)