
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict 
import argparse 
import json
import sys
# importing libraries required 


def arg_parser():
    ''' 
    Creates the argument parser and returns the parsed arguments
    Args: None
    Returns: Parsed arguments
    '''
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('arg1', nargs='?', default='flower', help='Description of arg1')
    parser.add_argument('--arch', type=str, default='densenet121', help='Choose the architecture')
    parser.add_argument('--learning_rate', type=float, default=1, help='Choose the learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Choose the number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Choose the number of epochs')
    parser.add_argument('--gpu', action="store_true", help='Choose the device')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Choose the directory to save the checkpoint')
    print(parser.parse_args())
    return parser.parse_args()



def make_dataloader(data_dir, train=True):
    ''' 
    Creates the dataloader for the data
    Args: data_dir, train
    Returns: Dataloader
    '''
    if train:
        transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])])
    image_datasets = datasets.ImageFolder(data_dir, transform=transform)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    return dataloaders, image_datasets


def label_mapping(file_path):
    ''' 
    Creates the mapping of the labels
    Args: file_path
    Returns: Mapping of the labels
    '''
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def make_model(arch, hidden_units, output_size):
    ''' 
    Creates the model according to the architecture 
    Args: arch, hidden_units
    Returns: Model

    '''
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Please choose from densenet121, vgg16 or alexnet")
        exit()
    for param in model.parameters():
        param.requires_grad = True
    if arch == 'densenet121':
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(hidden_units, output_size)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    elif arch == 'vgg16':
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(hidden_units, output_size)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    elif arch == 'alexnet':
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(9216, hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(hidden_units, output_size)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier
    return model



def save_model(model, save_directory, epochs,train_dataloader, arch, hidden_units, lr ):
    ''' 
    Saves the model
    Args: model, save_directory, epochs, optimizer, criterion , train_dataloader, arch, hidden_units, lr
    Returns: None
    '''
    model.class_to_idx = train_dataloader.dataset.class_to_idx
    checkpoint = {'arch': arch,
                  'hidden_units': hidden_units,
                  'epochs': epochs,
                  'lr': lr,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier}
    torch.save(checkpoint, save_directory)
    
               



def train_model(arch,model ,device ,epochs , lr,  train_dataloader, valid_dataloader,hidden_units, save_directory=None, reduce_lr = False):
    '''this function trains the model and saves the checkpoint
    Args: model, epochs, lr, train_dataloader, valid_dataloader, save_directory, reduce_lr
    Returns: model

    '''
    criterion = nn.NLLLoss()
    optimizer = optim.Adadelta(model.classifier.parameters(), lr)
    steps = 0
    running_loss = 0
    print_every = 10
    best_accuracy = 0
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False 
    for param in model.classifier.parameters():
        param.requires_grad = True
    print("Training started")
    for epoch in range(epochs):
        if epoch%4 == 0 and epoch != 0 and reduce_lr and optimizer.param_groups[0]['lr']>0.002:
             optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2

        for inputs, labels in train_dataloader:
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in valid_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                if save_directory!=None and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    save_model(arch=arch,model=model,save_directory=save_directory,epochs=epochs,train_dataloader=train_dataloader,hidden_units=hidden_units,lr=lr )
                    print("Model saved with accuracy {}".format(accuracy/len(valid_dataloader)))
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(valid_dataloader):.3f}.. "
                        f"Valid accuracy: {accuracy/len(valid_dataloader):.3f}")
                running_loss = 0
                model.train()
    print("Training complete")
    return model


def test_model(test_loader , model, device):
    ''' 
    Tests the model
    Args: test_loader, model
    Returns: None
    '''
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 102 test images: %d %%' % (
        100 * correct / total))

            

def main():
    args = arg_parser()
    data_dir = args.arg1
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    dataloader={}
    dataloader['train'] = make_dataloader(train_dir)
    dataloader['valid'] = make_dataloader(valid_dir, train=False)
    dataloader['test'] = make_dataloader(test_dir, train=False)
    class_to_idx = dataloader['train'].dataset.class_to_idx
    model = make_model(arch=args.arch, hidden_units=args.hidden_units, output_size=len(class_to_idx))
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model = train_model(arch=args.arch,model=model,device=device,epochs=args.epochs,lr=args.learning_rate,train_dataloader=dataloader['train'],valid_dataloader=dataloader['valid'],hidden_units=args.hidden_units,save_directory=args.save_dir,reduce_lr=True)
    model.to(device)
    test_model(test_loader=dataloader['test'],model=model, device=device)
    print("Done")


if __name__ == '__main__':
    main()
    