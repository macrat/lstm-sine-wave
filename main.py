import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy
import seaborn


class Model(chainer.Chain):
    def __init__(self):
        super().__init__(
            lstm=L.LSTM(1, 12),
            l1=L.Linear(12, 6),
            l2=L.Linear(6, 1),
        )

        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self)

    def predict(self, x):
        result = []
        for v in x:
            h = self.lstm(v.reshape((1, 1)))
            h = self.l1(h)
            h = self.l2(h)
            result.append(h.data[0][0])
        return numpy.array(result)

    def fit(self, x, y):
        self.reset()
        h = self.predict(x)

        result = []
        loss = 0
        for v, u in zip(x, y):
            h = self.lstm(v.reshape((1, 1)))
            h = self.l1(h)
            h = self.l2(h)
            result.append(h.data[0][0])
            loss += F.mean_squared_error(h, numpy.array([[u]]))

        self.zerograds()
        loss.backward()
        self.optimizer.update()

        return numpy.array(result), loss.data / len(result)

    def reset(self):
        self.lstm.reset_state()


if __name__ == '__main__':
    x_data = numpy.linspace(0, numpy.pi*2*10, 1000, dtype=numpy.float32)
    long_data = numpy.sin(x_data)
    data = long_data[:100]

    loss_log = [[], []]
    oldfig = None

    m = Model()

    for i in range(9999):
        predict, loss = m.fit(data[:-1], data[1:])
        print(i, loss)
        loss_log[0].append(loss)

        m.reset()
        xs = [0.0]
        for _ in range(999):
            xs.append(m.predict(numpy.array([xs[-1]], dtype=numpy.float32))[0])
        xs = numpy.array(xs, dtype=numpy.float32)

        xs_loss = float(F.mean_squared_error(xs, long_data).data)
        loss_log[1].append(xs_loss)

        fig = plt.figure(figsize=(16, 9))

        ax = plt.subplot(2, 2, 1)
        ax.set_ylim(-1.1, 1.1)
        ax.plot(predict, label='model(sin $x$)')
        ax.plot(data[1:], ':', c='orange', label='sin $x$')
        ax.text(-1, -1, 'loss={}'.format(loss), ha='left', va='bottom')
        ax.legend(loc='upper right')

        ax = plt.subplot(2, 2, 2)
        ax.set_ylim(0, None)
        ax.plot(loss_log[0], label='loss of model(sin $x$})')
        ax.plot(loss_log[1], c='green', label='loss of model($y_{x-1}$)')
        ax.legend(loc='upper right')

        ax = plt.subplot(2, 1, 2)
        ax.set_ylim(-1.1, 1.1)
        ax.plot(xs, c='green', label='model($y_{x-1}$); $y_0 = 0$')
        ax.plot(long_data, ':', c='orange', label='sin $x$')
        ax.text(-1, -1, 'loss={}'.format(xs_loss), ha='left', va='bottom')
        ax.legend(loc='upper right')

        fig.savefig('out/{0:04d}.png'.format(i))
        plt.pause(1)

        if oldfig is not None:
            plt.close(oldfig)
        oldfig = fig
