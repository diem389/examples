from main import Net, test
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch MNIST Load Test')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--model', default='./mnist.dat')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Current Device:", device)

    model = Net()
    model.load_state_dict(torch.load(args.model, map_location=device))
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    test(args, model, device, dataloader_kwargs)
