# %%
import torch
from torchvision import transforms
from dataset import MyDataset, TransformMnist


# #%%
def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    base_folder = "/Users/nielsraunkjaer/Desktop/Sem8/02476/dtu_mlops/data/corruptmnist/"

    train_img0 = torch.load(base_folder + "train_images_0.pt")
    train_img1 = torch.load(base_folder + "train_images_1.pt")
    train_img2 = torch.load(base_folder + "train_images_2.pt")
    train_img3 = torch.load(base_folder + "train_images_3.pt")
    train_img4 = torch.load(base_folder + "train_images_4.pt")
    train_img5 = torch.load(base_folder + "train_images_5.pt")
    all_train_img = torch.concat(
        [
            train_img0,
            train_img1,
            train_img2,
            train_img3,
            train_img4,
            train_img5,
        ],
        dim=0,
    )

    train_lbl0 = torch.load(base_folder + "train_target_0.pt")
    train_lbl1 = torch.load(base_folder + "train_target_1.pt")
    train_lbl2 = torch.load(base_folder + "train_target_2.pt")
    train_lbl3 = torch.load(base_folder + "train_target_3.pt")
    train_lbl4 = torch.load(base_folder + "train_target_4.pt")
    train_lbl5 = torch.load(base_folder + "train_target_5.pt")
    all_train_lbl = torch.concat(
        [
            train_lbl0,
            train_lbl1,
            train_lbl2,
            train_lbl3,
            train_lbl4,
            train_lbl5,
        ],
        dim=0,
    )

    test_img = torch.load(base_folder + "test_images.pt")
    test_lbl = torch.load(base_folder + "test_target.pt")

    # normalize_train = transforms.Normalize(mean=torch.mean(all_train_img), std=torch.std(all_train_img))
    # normalize_test = transforms.Normalize(mean=torch.mean(test_img), std=torch.std(test_img))

    # def transform_train(x):
    #     return normalize_train(x).view(784)
    # def transform_test(x):
    #     return normalize_test(x).view(784)

    transform_train = TransformMnist(all_train_img)
    transform_test = TransformMnist(test_img)

    train = MyDataset(all_train_img, all_train_lbl, transform=transform_train)
    test = MyDataset(test_img, test_lbl, transform=transform_test)

    return train, test


if __name__ == "__main__":
    train, test = mnist()
    torch.save(train.get_saveable_repr(), "./data/processed/train_data.pt")
    torch.save(train.get_saveable_repr(), "./data/processed/test_data.pt")
