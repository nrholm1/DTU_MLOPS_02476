from mlops_mnist_classifier.models.model import MyAwesomeModel
import torch

def test_model():
    model = MyAwesomeModel()
    example_input = torch.Tensor(1,28,28)

    log_ps,_ = model(example_input)

    assert log_ps.shape == torch.Size([1,10])