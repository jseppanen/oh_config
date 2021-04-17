"""
Adapted from the following Thinc example:
https://github.com/explosion/thinc/blob/master/examples/03_pos_tagger_basic_cnn.ipynb
"""

from thinc.api import Model, chain, strings2arrays, with_array, HashEmbed, expand_window, Relu, Softmax
import oh
import ml_datasets
from tqdm.notebook import tqdm
from thinc.api import fix_random_seed


def train_model(model, optimizer, n_iter, batch_size):
    (train_X, train_y), (dev_X, dev_y) = ml_datasets.ud_ancora_pos_tags()
    model.initialize(X=train_X[:5], Y=train_y[:5])
    for n in range(n_iter):
        loss = 0.0
        batches = model.ops.multibatch(batch_size, train_X, train_y, shuffle=True)
        for X, Y in tqdm(batches, leave=False):
            Yh, backprop = model.begin_update(X)
            d_loss = []
            for i in range(len(Yh)):
                d_loss.append(Yh[i] - Y[i])
                loss += ((Yh[i] - Y[i]) ** 2).sum()
            backprop(d_loss)
            model.finish_update(optimizer)
        score = evaluate(model, dev_X, dev_y, batch_size)
        print(f"{n}\t{loss:.2f}\t{score:.3f}")


def evaluate(model, dev_X, dev_Y, batch_size):
    correct = 0
    total = 0
    for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
        Yh = model.predict(X)
        for yh, y in zip(Yh, Y):
            correct += (y.argmax(axis=1) == yh.argmax(axis=1)).sum()
            total += y.shape[0]
    return float(correct / total)


@oh.register
def cnn_tagger(width: int, vector_width: int, nr_classes: int = 17):
    with Model.define_operators({">>": chain}):
        model = strings2arrays() >> with_array(
            HashEmbed(nO=width, nV=vector_width, column=0)
            >> expand_window(window_size=1)
            >> Relu(nO=width, nI=width * 3)
            >> Relu(nO=width, nI=width)
            >> Softmax(nO=nr_classes, nI=width)
        )
    return model


# XXX interpolation is not implemented yet
CONFIG = """
[hyper_params]
width = 32
vector_width = 16
learn_rate = 0.001

[training]
n_iter = 10
batch_size = 128

[model]
@call = cnn_tagger
width = 32  # ${hyper_params:width}
vector_width = 16  # ${hyper_params:vector_width}
nr_classes = 17

[optimizer]
@call = thinc.api/Adam
learn_rate = 0.001  # ${hyper_params:learn_rate}
"""

fix_random_seed(0)

oh.config.load_str(CONFIG)

model = oh.config.model()
optimizer = oh.config.optimizer()
n_iter = oh.config.training["n_iter"]
batch_size = oh.config.training["batch_size"]
train_model(model, optimizer, n_iter, batch_size)
