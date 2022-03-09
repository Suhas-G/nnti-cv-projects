import torch
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_cifar10(testdataset, filepath = "./path/to/model.pth.tar", debug = False):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 10]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    model_data = torch.load(filepath)
    model_width = int(model_data['params']['model_width'])
    model_depth = int(model_data['params']['model_depth'])
    model  = WideResNet(model_depth, 10, widen_factor=model_width)
    model.load_state_dict(model_data['state_dict'])
    model = model.to(DEVICE)
    model.eval()

    test_batch_size = model_data['params']['test_batch_size']
    dataloader = DataLoader(testdataset, batch_size = test_batch_size, shuffle = False)
    outputs = []
    count = 0
    avg_accuracy = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = F.softmax(model(data), dim=1)
            if debug:
                avg_accuracy += accuracy(output, target)[0]
            outputs.append(output.cpu())
            count += 1

    if debug:
        print('Average accuracy:', avg_accuracy/count)

    return torch.cat(outputs, dim=0)

def test_cifar100(testdataset, filepath="./path/to/model.pth.tar", debug = False):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 100]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''
    model_data = torch.load(filepath)
    model_width = int(model_data['params']['model_width'])
    model_depth = int(model_data['params']['model_depth'])
    model  = WideResNet(model_depth, 100, widen_factor=model_width)
    model.load_state_dict(model_data['state_dict'])
    model = model.to(DEVICE)
    model.eval()

    test_batch_size = model_data['params']['test_batch_size']
    dataloader = DataLoader(testdataset, batch_size = test_batch_size, shuffle = False)
    outputs = []
    count = 0
    avg_accuracy = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = F.softmax(model(data), dim=1)
            if debug:
                avg_accuracy += accuracy(output, target)[0]
            outputs.append(output.cpu())
            count += 1

    if debug:
        print('Average accuracy:', avg_accuracy/count)

    return torch.cat(outputs, dim=0)

if __name__ == "__main__":
    # Testing
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")

    # Dataset parameters
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")


    parser.add_argument("--model-path", type=str, required=True)
    
    args = parser.parse_args()

    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
        # Test the model
        output = test_cifar10(test_dataset, args.model_path, debug = True)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)

        # Test the model
        output = test_cifar100(test_dataset, args.model_path)

    print('Output shape: ', output.shape)

