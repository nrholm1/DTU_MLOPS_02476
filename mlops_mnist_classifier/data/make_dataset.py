# %%
import torch
from torchvision import transforms
from dataset import MyDataset, TransformMnist


# #%%
def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    base_folder = "/Users/nielsraunkjaer/Desktop/Sem8/02476.nosync/mlops_mnist_classifier/data/raw/corruptmnist_v2/"

    train_img0 = torch.load(base_folder + "train_images_0.pt")
    train_img1 = torch.load(base_folder + "train_images_1.pt")
    train_img2 = torch.load(base_folder + "train_images_2.pt")
    train_img3 = torch.load(base_folder + "train_images_3.pt")
    train_img4 = torch.load(base_folder + "train_images_4.pt")
    train_img5 = torch.load(base_folder + "train_images_5.pt")
    train_img6 = torch.load(base_folder + "train_images_6.pt")
    train_img7 = torch.load(base_folder + "train_images_7.pt")
    train_img8 = torch.load(base_folder + "train_images_8.pt")
    train_img9 = torch.load(base_folder + "train_images_9.pt")
    all_train_img = torch.concat(
        [
            train_img0,
            train_img1,
            train_img2,
            train_img3,
            train_img4,
            train_img5,
            train_img6,
            train_img7,
            train_img8,
            train_img9,
        ],
        dim=0,
    )

    print(f"shape of all_train_img: {all_train_img.shape}")

    train_lbl0 = torch.load(base_folder + "train_target_0.pt")
    train_lbl1 = torch.load(base_folder + "train_target_1.pt")
    train_lbl2 = torch.load(base_folder + "train_target_2.pt")
    train_lbl3 = torch.load(base_folder + "train_target_3.pt")
    train_lbl4 = torch.load(base_folder + "train_target_4.pt")
    train_lbl5 = torch.load(base_folder + "train_target_5.pt")
    train_lbl6 = torch.load(base_folder + "train_target_6.pt")
    train_lbl7 = torch.load(base_folder + "train_target_7.pt")
    train_lbl8 = torch.load(base_folder + "train_target_8.pt")
    train_lbl9 = torch.load(base_folder + "train_target_9.pt")
    all_train_lbl = torch.concat(
        [
            train_lbl0,
            train_lbl1,
            train_lbl2,
            train_lbl3,
            train_lbl4,
            train_lbl5,
            train_lbl6,
            train_lbl7,
            train_lbl8,
            train_lbl9,
        ],
        dim=0,
    )

    test_img = torch.load(base_folder + "test_images.pt")
    test_lbl = torch.load(base_folder + "test_target.pt")

    print(f"shape of test_img: {test_img.shape}")

    transform_train = TransformMnist(all_train_img)
    transform_test = TransformMnist(test_img)

    train = MyDataset(all_train_img, all_train_lbl, transform=transform_train)
    test = MyDataset(test_img, test_lbl, transform=transform_test)

    return train, test


if __name__ == "__main__":
    train, test = mnist()
    torch.save(train.get_saveable_repr(), "./data/processed/train_data.pt")
    torch.save(test.get_saveable_repr(), "./data/processed/test_data.pt")
