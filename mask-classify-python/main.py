from demo import demo
from model import train_model
import tensorflow.compat.v1 as tf

flags =  tf.app.flags
flags.DEFINE_string('MODE', 'train', 
                    'Set program to run in different mode, include train, valid and demo.')
flags.DEFINE_string('checkpoint_dir', './models', 
                    'Path to model file.')
flags.DEFINE_string('train_data', './data/fer2013/fer2013.csv',
                    'Path to training data.')
# flags.DEFINE_string('valid_data', './valid_sets/',
#                     'Path to training data.')
flags.DEFINE_boolean('show_box', False, 
                    'If true, the results will show detection box')
FLAGS = flags.FLAGS

def main() :
    assert FLAG.MODE in ('train', 'valid', 'demo')

    if FLAGS.MODE == 'demo':
        demo(FLAGS.checkpoint_dir, FLAGS.show_box)
    elif FLAGS.MODE == 'train':
        train_model(FLAGS.train_data)


if __name__ == '__main__':
    main()
