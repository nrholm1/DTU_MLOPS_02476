from matplotlib.colors import ListedColormap
import numpy as np, torch, click, matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from data.dataset import MyDataset, TransformMnistProcessed


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command
@click.argument("model_checkpoint")
@click.argument("predict_path")
def predict(model_checkpoint, predict_path):
    """Perform prediction on a given set of 28x28 images in a .npy-file"""
    model = torch.load(model_checkpoint)
    test_imgs = torch.Tensor(np.load(predict_path))

    log_ps, _ = model(test_imgs.unsqueeze(1))
    ps = torch.exp(log_ps)        
    top_p, top_class = ps.topk(1, dim=1)
    print(top_class)


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    imgs, lbls = torch.load("data/processed/test_data.pt")
    test_set = MyDataset(imgs, lbls, transform=TransformMnistProcessed(imgs))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    with torch.no_grad():
        model.eval()
        accuracy = []
        errors, total = 0, 0
        for images, labels in testloader:
            log_ps, _ = model(images.unsqueeze(1))
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            acc = torch.mean(equals.type(torch.FloatTensor))
            accuracy.append(acc.item() * 100)
            errors += len(labels) - torch.sum(equals)
            total += len(labels)
        print(f"Accuracy: {torch.mean(torch.Tensor(accuracy))}%")
        print(f"Errors/Total: {errors}/{total}")


@click.command()
@click.argument("model_checkpoint")
def visualize(model_checkpoint):
    """Visualize features extracted by a trained model."""
    print("Visualizing features")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    imgs, lbls = torch.load("data/processed/test_data.pt")
    test_set = MyDataset(imgs, lbls, transform=TransformMnistProcessed(imgs))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    with torch.no_grad():
        model.eval()
        tsne_features = None
        incl_labels = []
        for i, (images, labels) in enumerate(testloader):
            if i >= 5000 // len(labels):
                break
            _, features = model(images.unsqueeze(1))
            if tsne_features is None:
                tsne_features = features
            else:
                tsne_features = torch.concat([tsne_features, features])
            incl_labels.append(labels)

        tsne = TSNE(n_components=2, random_state=42)
        _tsne_features = tsne.fit_transform(tsne_features)

        # plt.scatter(_tsne_features[:, 0], _tsne_features[:, 1], c=labels, cmap='viridis', alpha=0.5)
        cmap = ListedColormap(['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'gray'])
        plt.scatter(_tsne_features[:, 0], _tsne_features[:, 1], c=incl_labels, cmap=cmap, alpha=0.5)
        plt.title("t-SNE Visualization of CNN Features")
        plt.colorbar()

        plt.savefig("reports/figures/cnn_tsne_plot.png")


cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(visualize)


if __name__ == "__main__":
    cli()
