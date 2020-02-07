from model import *
from data import *

import argparse

#from torchviz import make_dot


SEED = 1234
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    x_train , y_train, x_test, y_test = generate_data()
    x_torch , y_torch = torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device)
    
    # --------- ARGS --------- 
    epochs = args.epochs
    batch_size = args.batch_size
    ctx_size = args.ctx_size
    lr = args.lr
    print_step = args.print_step

    epsilon = 0.01

    # --------- MODEL --------- 
    x_dim = 1
    y_dim = 1
    r_dim = 128  #128

    encoder = Encoder(x_dim, y_dim, r_dim).to(device)
    decoder = Decoder(x_dim, r_dim, y_dim).to(device)
    net = CNPs(encoder=encoder, decoder=decoder).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr = lr) #0.01 and 10000 epochs!
    
    # --------- TRAIN --------- 
    for epoch in range(epochs):
        nll_loss =  0.0
        optimizer.zero_grad()
        
        # TODO -- Train model
        


        # -------

        if epoch % print_step == 0:
            print('Epoch', epoch, ': nll loss', nll_loss.item())
        #dot = make_dot(mean)
        #dot.render("model.png")

        nll_loss.backward()
        optimizer.step()

    print("final loss : nll loss", nll_loss.item())
    result_mean, result_var = None, None
    
    # TODO -- Calculate result_mean and result_var using your model


    # -------

    draw_graph(x_test, y_test, x_train, y_train, result_mean, np.sqrt(result_var))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep ensemble')
    parser.add_argument('--epochs',type=int,default=10000)
    parser.add_argument('--batch_size',type=int,default=20)
    parser.add_argument('--ctx_size',type=int,default=20)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--print_step', type=int, default=1000)

    args = parser.parse_args()
    main(args)