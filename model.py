import os
import tensorflow as tf

from nn import MLP, Shallow_CNN, Deep_CNN


class Model():

  def __init__(self, model_name):
    tf.reset_default_graph()

    # Build Graph
    self.model = self._init_model(model_name)
    self.prediction = tf.argmax(self.model.logits, axis=1)
    self.softmax = tf.nn.softmax(self.model.logits, axis=1)

    # Create a session
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    self.sess = tf.Session(config=session_conf)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    checkpoint_dir = os.path.join('checkpoints', model_name)
    print('\nLoading {} model from {}\n'.format(model_name, checkpoint_dir))
    saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
    print("Model loaded!")

  def _init_model(self, model_name):
    # Select the model
    if model_name == 'mlp':
      model = MLP()
    elif model_name == 'shallow':
      model = Shallow_CNN()
    elif model_name == 'deep':
      model = Deep_CNN()
    else:
      raise ValueError('--model should be "shallow" or "deep"')

    return model

  def predict(self, img):
    class_idx, probs = self.sess.run([self.prediction, self.softmax],
                                     feed_dict={self.model.x: img,
                                                self.model.is_training: False})
    labels = {0: 'sad', 1: 'happy'}
    return labels[class_idx[0]], probs[0][class_idx[0]]
