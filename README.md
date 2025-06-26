# MNIST Neural Networks

Two different approaches to digit classification - comparing PyTorch vs building everything from scratch with NumPy.

## About

This project tackles the classic MNIST handwritten digit classification problem using two different approaches. The first implementation uses PyTorch for a straightforward deep learning solution, while the second builds a neural network from the ground up using only NumPy to really understand what's happening under the hood.

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels.

## Files

- `mnsit-torch.py` - PyTorch implementation with a 3-layer neural network
- `mnsit-scratch.py` - From-scratch implementation using pure NumPy

## Setup

For the PyTorch version:
```bash
pip install torch torchvision matplotlib
```

For the NumPy version:
```bash
pip install numpy matplotlib
```

Optional (helps with dataset loading):
```bash
pip install tensorflow scikit-learn
```

## Running the Code

**PyTorch version:**
```bash
python mnsit-torch.py
```

**NumPy version:**
```bash
python mnsit-scratch.py
```

## What Each Implementation Does

### PyTorch Implementation
- Uses a fully connected network: 784 → 512 → 256 → 10 neurons
- ReLU activations, Adam optimizer, cross-entropy loss
- Trains for 10 epochs with batch size of 64
- Gets around 97-98% test accuracy
- Saves the trained model as `mnist_model.pth`

### From-Scratch Implementation  
- Configurable architecture (default: 784 → 128 → 64 → 10)
- Implements forward/backward propagation manually
- Uses Xavier initialization and mini-batch gradient descent
- More detailed training output and visualizations
- Gets around 85-92% test accuracy
- Shows per-class accuracy breakdown

## Results

The PyTorch version is faster and more accurate, which makes sense since it's using optimized implementations. The from-scratch version takes longer but really helps you understand how neural networks actually work.
with the pytorch model showing a remarkable accuracy of 97.43 and the model built from scratch showing an accuracy of 96.5%

Both versions show training progress and create plots of the loss/accuracy curves.
## Architecture Details

**PyTorch model:**
```
Input: 784 (28x28 flattened image)
Hidden 1: 512 neurons + ReLU
Hidden 2: 256 neurons + ReLU  
Output: 10 neurons (digit classes)
```

**NumPy model:**
```
Input: 784 (28x28 flattened image)
Hidden 1: 128 neurons + ReLU
Hidden 2: 64 neurons + ReLU
Output: 10 neurons + Softmax
```

## Key Learnings

Building the NumPy version from scratch really helped me understand:
- How backpropagation actually works mathematically
- Why proper weight initialization matters
- The difference between different activation functions
- How gradient descent optimizes the network

The PyTorch version shows how much easier modern frameworks make things, but understanding the fundamentals first made the framework approach much clearer.

## Customization

You can easily modify the network architectures by changing the layer sizes in either implementation. The from-scratch version is especially good for experimenting with different configurations to see how they affect performance.

## Issues I Ran Into

- Dataset loading can be tricky - the NumPy version includes fallback synthetic data if the real MNIST won't download
- Training the from-scratch version takes much longer, so start with fewer epochs when testing
- Memory usage can get high with large batch sizes

## Future Ideas

- Add convolutional layers for better image processing
- Implement dropout for regularization  
- Try different optimizers in the from-scratch version
- Add data augmentation to improve generalization

## Notes

This was my attempt at really understanding neural networks by implementing them two ways. The PyTorch version was pretty straightforward once I got the hang of the framework, but building everything from scratch with NumPy was definitely more challenging and time-consuming.

The from-scratch implementation really made me appreciate how much PyTorch handles automatically - all the gradient calculations, optimized matrix operations, and GPU acceleration. But doing it manually first gave me a much better intuition for what's actually happening during training.

Both approaches have their place - PyTorch for getting things done efficiently, and the manual implementation for really understanding the fundamentals.



