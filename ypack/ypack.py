import sys
import shutil
import copy
import logging
import tensorflow as tf
from contextlib import contextmanager
from tqdm import tqdm
import numpy as np
import itertools as itt
import functools as fct


def tf_print(a):
    return tf.Print(a, [tf.reduce_mean(a)])


def variable_by_name_end(name):
    vs = tf.get_collection('variables')
    for v in vs:
        if v.name.endswith(name):
            return v
    for v in vs:
        print(v.name)
    raise Exception('Variable {} not found'.format(name))

def variable_by_name(name):
    vs = tf.get_collection('variables')
    for v in vs:
        if v.name == name:
            return v
    for v in vs:
        print(v.name)
    raise Exception('Variable {} not found'.format(name))


class Callback:
    def __init__(self, steps_per_epoch=1):
        self.steps_per_epoch = steps_per_epoch

    def setup(self, trainer):
        self.trainer = trainer
        self._setup()

    def _setup(self):
        pass

    def before_init(self):
        self._before_init()

    def _before_init(self):
        pass

    def is_epoch_now(self):
        if self.steps_per_epoch <= 0:
            return False
        return self.trainer.step_count % self.steps_per_epoch == 0

    def trigger_epoch(self):
        self._trigger_epoch()

    def _trigger_epoch(self):
        pass

    def trigger_step(self):
        self._trigger_step()
        if self.is_epoch_now():
            self.trigger_epoch()

    def _trigger_step(self):
        pass


class ArgvPrinter(Callback):
    def _trigger_epoch(self):
        logging.info(" ".join(sys.argv))


class ModelSaver(Callback):
    def __init__(self, write_meta_graph=True, **kwargs):
        super().__init__(**kwargs)
        self.write_meta_graph = write_meta_graph

    def _setup(self):
        self.saver = tf.train.Saver()

    def _trigger_epoch(self):
        logging.info('Saving model')
        self.saver.save(self.trainer.sess, './logs/model.ckpt', write_meta_graph=self.write_meta_graph)


class ModelRestorer(Callback):
    def __init__(self, path, var_prefix=None):
        super().__init__()
        self.path = path
        self.prefix = var_prefix

    def _setup(self):
        self._to_restore = tf.global_variables()
        if self.prefix:
            self._to_restore = [v for v in self._to_restore if v.name.startswith(self.prefix)]
        self.saver = tf.train.Saver(self._to_restore)

    def _before_init(self):
        self.saver.restore(self.trainer.sess, self.path)
        self.trainer.init_op = tf.group(
                tf.local_variables_initializer(), 
                tf.variables_initializer([v for v in tf.global_variables() if v not in self._to_restore])
            )


class EpochCounter(Callback):
    def __init__(self, epochs, total_epochs=None):
        super().__init__(epochs)
        self.total_epochs = total_epochs

    def _setup(self):
        self.epochs = 0

    def _trigger_epoch(self):
        logging.info('Epoch: {}'.format(self.epochs) + ('' if self.total_epochs is None else ' of {}'.format(self.total_epochs)))
        self.epochs += 1


class StepDisplay(Callback):
    def _create_pbar(self):
        self.pbar = tqdm(total=self.steps_per_epoch)

    def _trigger_step(self):
        if self.is_epoch_now():
            if hasattr(self, 'pbar'):
                self.pbar.close()
            self._create_pbar()
        self.pbar.update(1)


class Evaluator:
    def setup(self, runner):
        self.runner = runner
        self._setup()

    def _setup(self):
        pass

    def _get_ops(self):
        return []

    def trigger_epoch(self, summary):
        self._trigger_epoch(summary)

    def _trigger_epoch(self, summary):
        pass


class EvalDatasetRunner(Callback):
    def __init__(self, steps_per_epoch, model, eval_dataset, evaluators=[]):
        super().__init__(steps_per_epoch)
        if model.feed:
            from tensorpack.dataflow.common import RepeatedData
            ds = eval_dataset
            ds.reset_state()
            self.data = ds
            ds = RepeatedData(ds, -1)
            self.data_producer = ds.get_data()
        else:
            self.data_producer = eval_dataset
        self.model = model
        self.evaluators = evaluators

    def _setup(self):
        if self.model.feed:
            self.data_queue = None
        else:
            self.data_queue = self.data_producer()

        with tf.variable_scope('', reuse=True), no_training_context():
            self.model.build_graph(self.data_queue)
        self.eval_ops = []
        for e in self.evaluators:
            e.setup(self)
            eops = e._get_ops()
            if eops is not None:
                self.eval_ops += eops
        with tf.control_dependencies(self.eval_ops):
            self.summary_op = tf.identity(tf.summary.merge_all(tf.GraphKeys.SUMMARIES))

    def _trigger_epoch(self):
        if self.trainer.step_count == 0:
            return
        if self.model.feed:
            batch = next(self.data_producer)
            feed = dict(zip(self.model.get_input_vars(), batch))
            summary_str = self.trainer.sess.run(self.summary_op, feed_dict=feed)
        else:
            summary_str = self.trainer.sess.run(self.summary_op)
        self.trainer.summary_writer.add_summary(summary_str, self.trainer.step_count + 1)
        self.trainer.summary_writer.flush()

        summ = tf.Summary()
        summ.ParseFromString(summary_str)
        for val in summ.value:
            if val.HasField('simple_value'):
                logging.info('{}: {}'.format(val.tag, val.simple_value))

        for e in self.evaluators:
            e.trigger_epoch(summ)

TRAINING_SUMMARY_KEY = 'training_summaries'


class Trainer:
    def __init__(self, model, data_producer, callbacks=[], write_train_summaries=True, train_data_size=None):
        self.model = model
        self.data_producer = data_producer
        self.callbacks = callbacks
        self.write_train_summaries = write_train_summaries
        self.train_data_size = train_data_size

    def setup(self):
        if self.model.feed:
            self.data_queue = None
        else:
            self.data_queue = self.data_producer()

        with training_context():
            self.model.build_graph(self.data_queue)
        self._setup_callbacks()
        self._setup()
        shutil.rmtree('./logs', ignore_errors=True)
        self.summary_writer = tf.summary.FileWriter('./logs')
        count_params()
        self.sess = tf.Session()
        self._before_init()
        print('----TRAINABLES----')
        for v in tf.trainable_variables():
            print(v)
        tf.get_default_graph().finalize()
        self.sess.run(self.init_op)

        if not self.model.feed:
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def _setup_callbacks(self):
        for c in self.callbacks:
            c.setup(self)

    def _before_init(self):
        for c in self.callbacks:
            c.before_init()

    def build_train_op(self, ops, global_step_op=None, training_summary_op=None, do_step=True):
        if global_step_op is None:
            gsv = get_global_step_var()
            if do_step:
                global_step_op = tf.assign_add(gsv, 1)
            else:
                global_step_op = tf.assign_add(gsv, 0)
        if training_summary_op is None:
            if do_step and self.write_train_summaries:
                training_summary_op = tf.summary.merge_all(TRAINING_SUMMARY_KEY)
            else:
                training_summary_op = tf.constant('', dtype=tf.string)
        train_op = tf.tuple((training_summary_op, global_step_op), control_inputs=[ops], name='train_op')
        return train_op

    def _get_train_ops(self):
        return [] 

    def _setup(self):
        tos = self._get_train_ops()
        main_to = self.build_train_op(tos[0], do_step=True)
        other_tos = [self.build_train_op(to, do_step=True) for to in tos[1:]]
        self.train_ops = itt.cycle([main_to] + other_tos)
        self.num_train_ops = len(other_tos) + 1
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def run_step(self):
        return self._run_step()

    def _run_step(self):
        if self.model.feed:
            batch = next(self.data_producer)
            feed = dict(zip(self.model.get_input_vars(), batch))
            summary_str, global_step = self.sess.run(next(self.train_ops), feed_dict=feed)
        else:
            summary_str, global_step = self.sess.run(next(self.train_ops))
        did_step = global_step == self.step_count + 1
        if self.write_train_summaries and len(summary_str) > 0 and did_step:
            self.summary_writer.add_summary(summary_str, global_step=global_step)
            self.summary_writer.flush()
        return did_step

    def _stop(self):
        pass

    def stop(self):
        if self.model.feed:
            return self._stop()
        return self._stop() or self.coord.should_stop()

    def run_callbacks(self):
        for c in self.callbacks:
            c.trigger_step()

    def train(self):
        with tf.Graph().as_default():
            self.setup()
            self.step_count = 0
            stop_it = False
            try:
                while not self.stop() and not stop_it:
                    try:
                        did_step = self.run_step()
                    except (StopIteration, tf.errors.OutOfRangeError):
                        stop_it = True
                        did_step = False
                    self.run_callbacks()
                    if did_step:
                        self.step_count += 1
            finally:
                if not self.model.feed:
                    self.coord.request_stop()
            if not self.model.feed:
                self.coord.join(self.threads)
            self.sess.close()


class ModelDesc:
    def __init__(self, feed=False):
        self.feed = feed

    def build_graph(self, input_vars=None):
        if input_vars is None:
            input_vars = self.get_input_vars()
        return self._build_graph(input_vars)

    def _build_graph(self, input_vars):
        pass

    def get_input_vars(self):
        if not hasattr(self.__class__, '_input_vars'):
            self.__class__._input_vars = self._get_input_vars()
        return self.__class__._input_vars

    def _get_input_vars(self):
        pass


GLOBAL_STEP_VAR_NAME = 'global_step_var:0'
GLOBAL_STEP_OP_NAME = 'global_step_var'


def get_global_step_var():
    """
    Returns:
        tf.Tensor: the global_step variable in the current graph. create if
        doesn't exist.
    """
    try:
        return tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    except KeyError:
        scope = tf.get_variable_scope()
        assert scope.name == '', \
            "Creating global_step_var under a variable scope would cause problems!"
        with tf.variable_scope(scope, reuse=False):
            var = tf.get_variable(GLOBAL_STEP_OP_NAME, shape=[],
                                  initializer=tf.constant_initializer(dtype=tf.int32),
                                  trainable=False, dtype=tf.int32)
        return var


def get_global_step(sess):
    """
    Returns:
        float: global_step value in current graph and session"""
    # return tf.train.global_step(
        # tf.get_default_session(),
        # get_global_step_var())
    return get_global_step_var().eval(session=sess)


_ypack_global_context = None


def get_global_context():
    global _ypack_global_context
    if _ypack_global_context is None:
        _ypack_global_context = dict()
    return _ypack_global_context


@contextmanager
def context(**kwargs):
    gc = get_global_context()
    c = copy.deepcopy(gc)
    gc.update(kwargs)
    yield
    gc.clear()
    gc.update(c)
    return


@contextmanager
def training_context(**kwargs):
    kwargs['training'] = True
    c = context().func(**kwargs)
    next(c)
    yield
    next(c)
    return


@contextmanager
def no_training_context(**kwargs):
    kwargs['training'] = False
    c = context().func(**kwargs)
    next(c)
    yield
    next(c)
    return


def context_var(key, default=None):
    return get_global_context().get(key, default)


def set_default_context(**kwargs):
    global _ypack_global_context
    if _ypack_global_context is not None:
        logging.warn("default context set twice")
    _ypack_global_context = kwargs


def is_training():
    return context_var('training', False)


def random_select(n, k, resample=False):
    if resample:
        sel = tf.random_uniform([k], maxval=n, dtype=tf.int32)
    else:
        r = np.arange(n, dtype=np.int32)
        s = tf.random_shuffle(r)
        sel = s[:k]
    return sel


def lrelu(x, alpha=0.2):
    xtop = tf.nn.relu(x)
    xbot = tf.nn.relu(-xtop)
    xx = xtop - tf.constant(alpha, dtype=tf.float32) * xbot
    return xx


#http://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
def count_params():
    "print number of trainable variables"
    size = lambda v: fct.reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size(v) for v in tf.trainable_variables())
    print("Model size: %dK" % (n//1000,))
