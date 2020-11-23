import tensorflow as tf
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(input_graph='./train.pbtxt',
                          input_saver="",
                          input_binary=False,
                          input_checkpoint='./models/mask_model-100',
                          output_node_names='Softmax',
                          restore_op_name="",
                          filename_tensor_name="",
                          output_graph='./freeze.pb',
                          clear_devices=False,
                          initializer_nodes="")