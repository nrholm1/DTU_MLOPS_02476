from mlops_mnist_classifier.data.dataset import MyDataset, TransformMnistProcessed
import torch


def test_data():
    train_imgs,train_lbls = torch.load("data/processed/train_data.pt")
    test_imgs,test_lbls = torch.load("data/processed/test_data.pt")
    
    train_dataset = MyDataset(train_imgs, train_lbls, transform=TransformMnistProcessed(train_imgs))
    test_dataset = MyDataset(test_imgs, test_lbls, transform=TransformMnistProcessed(test_imgs))
    
    N_train = 50_000
    N_test = 5_000
    assert len(train_dataset) == N_train 
    assert len(test_dataset) == N_test
    
    train_imgs_shape = train_imgs[0].shape
    test_imgs_shape = test_imgs[0].shape
    assert train_imgs_shape == torch.Size([1,28,28]) or train_imgs_shape == torch.Size([784])
    assert test_imgs_shape == torch.Size([1,28,28]) or test_imgs_shape == torch.Size([784])
    
    all_unique_targets = torch.Tensor([i for i in range(10)])
    assert torch.sort(torch.unique(train_lbls)) == all_unique_targets
    assert torch.sort(torch.unique(test_lbls)) == all_unique_targets