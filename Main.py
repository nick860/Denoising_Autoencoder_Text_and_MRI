import Autoencoder
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np

def process_image(img_path):
    """
    param img_path: Path to the image
    return: Processed image
    """
    img = Image.open(img_path)
    img = img.resize((300, 300))
    img = np.array(img)
    img = img / 255
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)  # Repeat grayscale image to have 3 channels
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()

def data_loader(train_path, test_path, label_path):
    """
    param train_path: Path to the training images
    param test_path: Path to the test images
    param label_path: Path to the label images
    return: DataLoader for the training and test images
    """
    # Set the transforms
    train = []
    train_cleaned = []
    test = []

    for f in sorted(os.listdir(label_path)):
        train.append(process_image(os.path.join(label_path, f)))

    for f in sorted(os.listdir(train_path)):
        train_cleaned.append(process_image(os.path.join(train_path, f)))

    train_loader = list(zip(train, train_cleaned))
    for f in sorted(os.listdir(test_path)):
        test.append(process_image(os.path.join(test_path, f)))

    return train_loader, test

def predict_image(model, img):
    """
    param model: Trained model
    param img: Image to predict
    return: Predicted image
    """
    model.eval()
    with torch.no_grad():
        output = model(img)
        return output

def epoch_train(model, train_loader, criterion, optimizer, device):
    """
    param model: Model to train
    param train_loader: DataLoader for the training images
    param criterion: Loss function
    param optimizer: Optimizer
    param device: Device to train the model
    return: Loss
    """
    model.train()
    #for batch in train_loader:
    counter = 0
    for (imgs, labels) in tqdm(train_loader, total= len(train_loader)): #len(train_loader)):
        imgs = imgs.unsqueeze(0) # Add batch dimension
        labels = labels.unsqueeze(0) # Add batch dimension
        imgs, labels = imgs.to(device), labels.to(device) # Move to GPU
        optimizer.zero_grad()
        outputs = model(imgs) # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        counter += 1
        if counter == 144:
            break
    return loss.item()

def training(model, train_loader, criterion, optimizer, device, epochs):
    """
    param model: Model to train
    param train_loader: DataLoader for the training images
    param criterion: Loss function
    param optimizer: Optimizer
    param device: Device to train the model
    param epochs: Number of epochs
    return: None
    """
    for epoch in range(epochs):
        loss = epoch_train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')

def predict_image(model, test, num_images=10):
    """
    param model: Trained model
    param test: Test images
    param num_images: Number of images to predict
    return: None
    """
    model.eval()
    for i in range(num_images):
        with torch.no_grad():
            print("From this image : ")
            img = test[i].permute(1, 2, 0)
            plt.imshow(img)
            plt.show()
            print("Into to this image : ")
            img_denoise = test[i].unsqueeze(0)
            img = model(img_denoise)
            img = np.array(img.squeeze(0).permute(1, 2, 0))
            plt.imshow(img)
            plt.show()

def train_the_model(train_loader):
      """
      param train_loader: DataLoader for the training images
      return: Trained model
      """
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model = Autoencoder.Autoencoder().to(device)
      criterion = nn.MSELoss()
      optimizer = optim.Adam(model.parameters(), lr=0.001)
      training(model, train_loader, criterion, optimizer, device, 1)
      # save the model
      torch.save(model.state_dict(), 'denoising_MRI.pth')
      return model

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'dataset2') 
    path_train = os.path.join(path, 'train_cleaned')
    path_label = os.path.join(path, 'train')
    path_test = os.path.join(path, 'test')
    train_loader, test = data_loader(path_train, path_test, path_label)
    #model = train_the_model(train_loader)
    # Load the trained model
    model = Autoencoder.Autoencoder()
    model.load_state_dict(torch.load('denoising_MRI.pth'))
    
    predict_image(model, test, num_images=100)


