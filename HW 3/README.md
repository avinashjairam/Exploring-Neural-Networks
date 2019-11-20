## Problem 2

display training and validation losses and accuracies for each training

(a) train CNN (make sure to apply dataset normalization and dropout)
- [x] (i) adam optimizer
- [x] (ii) SGD with nesterov momentum

(b) choose best model from (a).
then
- [x] (i) add batch normalization
- [x] (ii) add data augmentation
- [x] (iii) add xavier initialization
- [x] (iv) vary dropout: ```0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8```
- [x] (v) add learning rate scheduler
- [ ] (vi) show confusion matrix for just (v)

(c) use the model with batch norm, data augmentation, xavier init, best dropout prob and learning rate scheduler, and

- [ ] (i) show four randomly augmented images from each class
- [ ] (ii) visualize activations of conv layer channels for two images
- [ ] (iii) visualize learned conv filters
