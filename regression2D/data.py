#-------------------------- 1D DATA ----------------------------
import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_data=20, x_range=(-3,3), std=3.):
    # train data    
    x_train = [[np.random.uniform(*x_range)] for _ in range(num_data)]
    y_train = [[x[0]**3 +np.random.normal(0,std)] for x in x_train]

    # test data
    x_test = np.linspace(-6,6,100).reshape(100,1) # test data for regression
    y_test = x_test**3

    return x_train ,y_train, x_test, y_test


def draw_graph(x,y,x_set,y_set,mean_predict,std): # x-s
    plt.plot(x,y,'b-', label = "Ground Truth")
    plt.plot(x_set, y_set,'ro', label = 'data points')
    plt.plot(x, mean_predict, label='MLPs (MSE)', color='grey')
    plt.fill_between(x.reshape(-1), (mean_predict-3*std).reshape(100,), (mean_predict+3*std).reshape(100,),color='grey',alpha=0.3)

    plt.legend()
    plt.savefig("result.png")
    #plt.show()



#-------------------------- 2D DATA ----------------------------
import torch
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_data_2d():
    # TRAIN & TEST MNIST IMAGES
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            torchvision.transforms.Lambda(lambda im: im.reshape(-1, 1))
            ])
    mnist_train = torchvision.datasets.MNIST("data", train=True, download=True, transform=transforms)
    mnist_test = torchvision.datasets.MNIST("data", train=False, download=True, transform=transforms)
    
    # VALIDATION IMAGES
    # Select 10 of each digit for validation
    indices = [i * 1000 + j for i in range(10) for j in range(10)]
    validation_set = torch.utils.data.Subset(mnist_test, indices)

    # TRAIN X DATA (grid)
    # Construct the image coordinate grid
    x = np.stack(np.meshgrid(np.linspace(-1.0, 1.0, 28), np.linspace(-1.0, 1.0, 28)), axis=2).reshape(-1, 2)
    x = torch.from_numpy(x).float()

    return mnist_train, mnist_test, validation_set, x


def plot_completion(net, img, x, y, path, name, n_pixels):
    npoints = [10, 50, 250, 500, 784]
    
    # Normalize img to [0..1]
    img = (img - img.min())
    img = img / img.max()
    
    fig, axes = plt.subplots(3, len(npoints), squeeze=False)
    
    for i, n in enumerate(npoints):
        mask = torch.zeros(n_pixels, dtype=torch.uint8)
        mask[torch.randperm(n_pixels)[:n]] = 1

        x_ctx = x[mask].to(device)
        y_ctx = y[mask].to(device)

        mu, sigma = net(x_ctx, y_ctx, x)
        mu = mu.cpu()
        sigma = sigma.cpu()

        mask = mask.numpy().astype(np.bool)
        
        # Show observed points
        masked_img = np.tile(img.clone().numpy()[:, :, np.newaxis], (1, 1, 3))
        masked_img[~mask] = [0.0, 0.0, 0.5]
        axes[0][i].imshow(masked_img.reshape(28, 28, 3))
        
        # Show the mean
        mean_img = np.clip(np.tile(mu.detach().numpy(), (1, 1, 3)), 0, 1)
        axes[1][i].imshow(mean_img.reshape(28, 28, 3))
        
        # Show the standard deviation
        std_img = np.clip(np.tile(sigma.detach().numpy(), (1, 1, 3)), 0, 1)
        axes[2][i].imshow(std_img.reshape(28, 28, 3))
        
        axes[0][i].set_title("{} observed".format(n))
        for j in range(3):
            axes[j][i].axis("off")
    
    #plt.show()
    plt.savefig(path + name)
    plt.cla()
