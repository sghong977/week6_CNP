from model import *
from data import *

import argparse

#from torchviz import make_dot
# python -W ignore::UserWarning main.py

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    # --------- ARGS --------- 
    epochs = args.epochs
    batch_size = args.batch_size
    ctx_size = args.ctx_size
    lr = args.lr
    print_step = args.print_step
    path = args.result_path
    wd = args.weight_decay

    epsilon = 0.01

    """ --------------- DATA ----------------
    x : complete grid (location of pixel)
    mnist_train, test, validation : intensity (mnist image data)

    x_ctx, y_ctx : given context (from train set)
    x_obs, y_obs : train data (see formula(4) in the paper.)
    """
    mnist_train, mnist_test, validation_set, x = generate_data_2d()
    x = x.to(device)
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    n_pixels = int(np.prod(mnist_train[0][0].size()))

    # Constant selection of 50 pixels to observe for validation
    val_mask = torch.zeros(n_pixels, dtype=torch.uint8)
    val_mask[torch.randperm(n_pixels)[:50]] = 1
    val_mask.to(device)

    
    # --------- MODEL --------- 
    x_dim = 2
    y_dim = 1
    r_dim = 128  #128

    encoder = Encoder(x_dim, y_dim, r_dim).to(device)
    decoder = Decoder(x_dim, r_dim, y_dim).to(device)
    net = CNPs(encoder=encoder, decoder=decoder).to(device)
    #print(net)

    #optimizer = torch.optim.AdamW(net.parameters(), lr = lr, weight_decay=wd) #0.01 and 10000 epochs!
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=0.9, weight_decay=wd) #0.01 and 10000 epochs!
    
    # --------- TRAIN --------- 
    for epoch in range(epochs):
        for batch, data in enumerate(mnist_train_loader):
            optimizer.zero_grad()

            # Iterate over the minibatch
            loss = 0.0
            imgs, _ = data
            
            # one image
            for y in imgs:
                y = torch.Tensor(y).to(device)

                # Select observed and target pixels at random
                mask = torch.zeros(n_pixels, dtype=torch.uint8).to(device)
                mask[torch.randperm(n_pixels)[:ctx_size]] = 1            # torch.randint(1, n_pixels, (1,), dtype=torch.long)
                
                x_ctx = x[mask].to(device)
                y_ctx = y[mask].to(device)
                x_obs = x.to(device)
                y_obs = y.to(device)
                            
                mu, var = net(x_ctx, y_ctx, x_obs)
                loss += NLLloss(y_obs, mu, var)
            
            # Take the mean over the minibatch
            loss /= imgs.shape[0]

            loss.backward()
            optimizer.step()

            #--------------- VALIDATION -------------------
            if (epoch == 0 and batch == 0) or batch in [149, 299, 449, 599]:
                val_llh = 0.0
                for y, _ in validation_set:
                    x_ctx = x[val_mask].to(device)
                    y_ctx = y[val_mask].to(device)
                    x_obs = x[~val_mask].to(device)
                    y_obs = y[~val_mask].to(device)
                    mu, var = net(x_ctx, y_ctx, x_obs)
                    val_llh += NLLloss(y_obs, mu, var)
                val_llh /= len(validation_set)
                
                print("Epoch {} Batch {}/{}: Loss {:.4f} - Val LLH {:.4f}".format(
                    1 + epoch, 1 + batch, len(mnist_train_loader), loss, val_llh))
                plot_index = int(np.random.randint(len(validation_set)))

                name = 'Epoch_'+str(1+epoch)+'_Batch_'+str(1+batch)+'.png'
                plot_completion(net, validation_set[plot_index][0], x, y, path, name, n_pixels)
            
            #torch.save(mnist_np.state_dict(), "checkpoint-{}-{}.pth.tar".format(epoch, batch))
   
    # END OF TRAIN
    print("final loss : nll loss", loss.item())
    result_mean, result_var = net(x_torch, y_torch, torch.FloatTensor(x_test).to(device)) #net(torch.FloatTensor(x).to(device))
    result_mean, result_var = result_mean.cpu().detach().numpy(), result_var.cpu().detach().numpy()
    draw_graph(x_test, y_test, x_train, y_train, result_mean, np.sqrt(result_var))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep ensemble')
    parser.add_argument('--epochs',type=int,default=10000)
    parser.add_argument('--batch_size',type=int,default=100)

    parser.add_argument('--ctx_size',type=int,default=250)

    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    parser.add_argument('--print_step', type=int, default=1000)
    parser.add_argument('--result_path', type=str, default='mnist_results/')

    args = parser.parse_args()
    main(args)