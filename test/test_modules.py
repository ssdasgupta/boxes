from boxes.box_wrapper import *
from boxes.modules import LSTMBox, PytorchSeq2BoxWrapper, mask_from_lens
import torch
import logging
import numpy as np
import pytest

logger = logging.getLogger(__name__)


def test_simple_forward_pass():
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    hidden_dim = 4
    inp_shape = (batch_size, seq_len, inp_dim)
    inp = torch.tensor(np.random.rand(*inp_shape)).float()
    lstm_boxes_layer = LSTMBox(inp_dim, hidden_dim, batch_first=True)
    boxes, _ = lstm_boxes_layer(inp)

    # boxes the usual way
    lstm = torch.nn.LSTM(inp_dim, hidden_dim, batch_first=True)

    lstm.load_state_dict(
        lstm_boxes_layer.state_dict()
    )  # loading state is necessary otherwise weights would be randomly initialized

    lstm_out, _ = lstm(inp)

    z = lstm_out.index_select(
        -1, torch.tensor(list(range(int(hidden_dim / 2))), dtype=torch.int64))
    Z = lstm_out.index_select(
        -1,
        torch.tensor(
            list(range(int(hidden_dim / 2), hidden_dim)), dtype=torch.int64))
    boxes2 = SigmoidBoxTensor.from_zZ(z, Z).data
    assert boxes.shape == boxes2.shape
    assert np.allclose(boxes.data.numpy(), boxes2.data.numpy())


def test_simple_forward_pass_errors():
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    hidden_dim = 3
    inp_shape = (batch_size, seq_len, inp_dim)
    inp = torch.tensor(np.random.rand(*inp_shape)).float()
    with pytest.raises(ValueError):
        lstm_boxes_layer = LSTMBox(inp_dim, hidden_dim, batch_first=True)


def test_simple_forward_pass_bidirectional():
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    hidden_dim = 4
    inp_shape = (batch_size, seq_len, inp_dim)
    inp = torch.tensor(np.random.rand(*inp_shape)).float()
    lstm_boxes_layer = LSTMBox(
        inp_dim, hidden_dim, batch_first=True, bidirectional=True)
    boxes, _ = lstm_boxes_layer(inp)


def test_packed_forward_pass():
    batch_size = 5
    inp_dim = 6
    hidden_dim = 4
    seq_lens = np.random.randint(1, 10, size=(batch_size, )).tolist()
    packed_inp = torch.nn.utils.rnn.pack_sequence(
        [torch.tensor(np.random.rand(l, inp_dim)) for l in seq_lens],
        enforce_sorted=False).float()
    lstm_boxes_layer = LSTMBox(inp_dim, hidden_dim, batch_first=True)
    boxes, _ = lstm_boxes_layer(packed_inp)
    assert tuple(boxes.shape) == (batch_size, max(seq_lens), 2,
                                  int(hidden_dim / 2))


def test_simple_grad():
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    hidden_dim = 4
    inp_shape = (batch_size, seq_len, inp_dim)
    lstm_boxes_layer = LSTMBox(inp_dim, hidden_dim, batch_first=True).double()

    def test_case():
        inp = torch.tensor(
            np.random.rand(*inp_shape), requires_grad=True).double()
        torch.autograd.gradcheck(lambda x: lstm_boxes_layer(x)[0], inp)

    for i in range(1):
        test_case()


def test_simple_forward_pass_seq2box():
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    hidden_dim = 4
    inp_shape = (batch_size, seq_len, inp_dim)
    inp = torch.tensor(np.random.rand(*inp_shape)).float()
    lstm = torch.nn.LSTM(inp_dim, hidden_dim, batch_first=True)
    lstm_boxes_layer = PytorchSeq2BoxWrapper(lstm)
    boxes = lstm_boxes_layer(inp)

    # boxes the usual way
    lstm2 = torch.nn.LSTM(inp_dim, hidden_dim, batch_first=True)

    lstm2.load_state_dict(
        lstm.state_dict()
    )  # loading state is necessary otherwise weights would be randomly initialized

    lstm_out, _ = lstm(inp)

    z = lstm_out.index_select(
        -1, torch.tensor(list(range(int(hidden_dim / 2))),
                         dtype=torch.int64))[:, -1, ...]
    Z = lstm_out.index_select(
        -1,
        torch.tensor(
            list(range(int(hidden_dim / 2), hidden_dim)),
            dtype=torch.int64))[:, -1, ...]
    boxes2 = SigmoidBoxTensor.from_zZ(z, Z).data
    assert boxes.data.shape == boxes2.shape
    assert np.allclose(boxes.data.data.numpy(), boxes2.data.numpy())


def test_simple_forward_pass_errors_seq2box():
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    hidden_dim = 3
    inp_shape = (batch_size, seq_len, inp_dim)
    inp = torch.tensor(np.random.rand(*inp_shape)).float()
    with pytest.raises(ValueError):
        lstm = torch.nn.LSTM(inp_dim, hidden_dim, batch_first=True)
        boxes_layer = PytorchSeq2BoxWrapper(lstm)


def test_simple_forward_pass_bidirectional_seq2box():
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    hidden_dim = 4
    inp_shape = (batch_size, seq_len, inp_dim)
    inp = torch.tensor(np.random.rand(*inp_shape)).float()
    lstm = torch.nn.LSTM(
        inp_dim, hidden_dim, batch_first=True, bidirectional=True)
    boxes_layer = PytorchSeq2BoxWrapper(lstm)
    boxes = boxes_layer(inp).data


def test_masked_forward_pass_seq2box():
    batch_size = 5
    inp_dim = 6
    hidden_dim = 4
    seq_lens = torch.tensor(
        np.random.randint(1, 10, size=(batch_size, )).tolist())
    inp = torch.tensor(
        np.random.rand(batch_size, int(seq_lens.max().item()),
                       inp_dim)).float()
    lstm = torch.nn.LSTM(inp_dim, hidden_dim, batch_first=True)
    boxes_layer = PytorchSeq2BoxWrapper(lstm)
    mask = mask_from_lens(seq_lens.detach().tolist())
    boxes = boxes_layer(inp, mask).data


#def test_simple_grad_seq2box():
#batch_size = 5
#seq_len = 3
#inp_dim = 10
#hidden_dim = 4
#inp_shape = (batch_size, seq_len, inp_dim)
#lstm_boxes_layer = LSTMBox(inp_dim, hidden_dim, batch_first=True).double()

#
#    def test_case():
#        inp = torch.tensor(
#            np.random.rand(*inp_shape), requires_grad=True).double()
#        torch.autograd.gradcheck(lambda x: lstm_boxes_layer(x)[0], inp)
#
#    for i in range(10):
#        test_case()
#
#
#def test_simple_grad_LSTM():
#    batch_size = 5
#    seq_len = 3
#    inp_dim = 10
#    hidden_dim = 4
#    inp_shape = (batch_size, seq_len, inp_dim)
#    lstm_boxes_layer = torch.nn.LSTM(
#        inp_dim, hidden_dim, batch_first=True).double()
#
#    def test_case():
#        inp = torch.tensor(
#            np.random.rand(*inp_shape), requires_grad=True).double()
#        torch.autograd.gradcheck(lambda inp: lstm_boxes_layer(inp)[0], inp)
#
#    for i in range(1):
#        test_case()

if __name__ == '__main__':
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    hidden_dim = 4
    inp_shape = (batch_size, seq_len, inp_dim)
    inp = torch.tensor(np.random.rand(*inp_shape), requires_grad=True).double()
    lstm_boxes_layer = LSTMSigmoidBox(
        inp_dim, hidden_dim, batch_first=True).double()
    #lstm_boxes_layer = torch.nn.LSTM(
    #    inp_dim, hidden_dim, batch_first=True).double()
    boxes = lstm_boxes_layer(inp)
    res = torch.sum(boxes)
    res.backward()
    print(inp.grad)
