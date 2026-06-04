# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Convolutional Autoencoder for Image Denoising

In real-world applications, images are often corrupted by noise due to factors like low lighting, sensor errors, or transmission issues. Noise reduces image quality and affects downstream tasks such as recognition, segmentation, and analysis.

The goal of this project is to develop a Convolutional Autoencoder (CAE) that can automatically learn to remove noise from images and reconstruct clean versions.

## DESIGN STEPS
### STEP 1: 

Understanding and Dataset Selection

### STEP 2: 

Preprocessing the Dataset

### STEP 3: 

 Design the Convolutional Autoencoder Architecture

### STEP 4: 

Compile and Train the Model

### STEP 5: 

 Evaluate the Model

### STEP 6: 
Visualization and Analysis



## PROGRAM

### Name: KANNAN N

### Register Number: 212223230097

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,output_padding=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,output_padding=1,padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x



# Initialize model
model =DenoisingAutoencoder().to(device)
criterion =nn.MSELoss()
optimizer =optim.Adam(model.parameters(),lr=1e-3)

# Training function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name:KANNAN N")
    print("Register number: 212223230097")
    for epoch in range(epochs):
      running_loss=0.0
      for images,_ in loader:
        images=images.to(device)
        noisy_images=add_noise(images).to(device)

        outputs=model(noisy_images)
        loss=criterion(outputs,images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
      print(f"Epoch [{epoch+1}/{epochs}],Loss: {running_loss/len(loader):.4f}")

# Visualization function
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:     KANNAN N            ")
    print("Register Number:      212223230097")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


```

### OUTPUT

### Model Summary
<img width="543" height="421" alt="image" src="https://github.com/user-attachments/assets/a760cadc-469b-4f95-9ea5-3ad714d7ece1" />



### Training loss
<img width="262" height="101" alt="image" src="https://github.com/user-attachments/assets/54978988-5da3-4030-94c2-e138612483ca" />



## Original vs Noisy Vs Reconstructed Image
<img width="762" height="278" alt="image" src="https://github.com/user-attachments/assets/b5d203c6-0219-4588-b69e-160192226838" />



## RESULT
Thus, develop a convolutional autoencoder for image denoising application excuted succesfully
