import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import  make_model
from torchvision import  transforms, models
def arg_parser():
    '''this function parses the arguments'''
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Point to impage file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Point to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.' , default=5)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store_true", dest="gpu")
    args = parser.parse_args()
    print(args)
    return args


def load_checkpoint(checkpoint_path):
    '''this function loads the checkpoint
    and returns the model
    Args : checkpoint_path
    Returns : model
    '''
    checkpoint = torch.load(checkpoint_path)
    model = make_model(checkpoint['arch'], checkpoint['hidden_units'], len(checkpoint['class_to_idx']))
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    '''this function processes the image and returns
    the processed image
    Args : image
    Returns : processed image
    '''
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
    img = PIL.Image.open(image)
    # convert to RGB
    image = img.convert('RGB')
    # resize
    image = transform(image)
    # add batch dimension
    image = image.unsqueeze(0)
    return image

def predict(image_path , model , top_k=5 , device = 'cpu', cat_to_name = 'cat_to_name.json'):
    '''this function predicts the class of the image
    and returns the top k probabilities
    Args : image_path , model , top_k
    Returns : top k probabilities
    '''
    # process image
    model.to(device)

    # set model to evaluation mode
    model.eval()

    #convert image to tensor
    torch_image = process_image(image_path)

    # move to device
    torch_image = torch_image.to(device)

    # calculate the class probabilities (softmax) for img
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    top_probs.to('cpu')
    top_labels.to('cpu')
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


    
def load_json_labels(path):
    '''this function loads the json file
    and returns the json file
    Args : path
    Returns : dict  '''

    with open (path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def print_probs(probs, flowers):
    '''this function prints the probabilities
    and the flowers
    Args : probs, flowers
    Returns : None
    '''
    for i in range(len(probs)):
        print("Flower: {} with a probability of: {}".format(flowers[i], probs[i]))

def main():

    args = arg_parser()
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    model= load_checkpoint(args.checkpoint)
    cat_to_name = load_json_labels(args.category_names)
    probs, labels, flowers = predict(image_path=args.image, model=model, top_k=args.top_k, device=device, cat_to_name=cat_to_name
                                     )
    print_probs(probs, flowers)

if __name__ == "__main__":
    main()