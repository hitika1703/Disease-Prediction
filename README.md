# Disease Prediction
🧠 Alzheimer’s Disease Prediction Using CNN and Pre-trained Models


📘 Overview  
This project focuses on the early detection and classification of Alzheimer’s disease using deep learning techniques, particularly Convolutional Neural Networks (CNNs). The goal is to automate the analysis of brain MRI images and classify them into different stages of dementia using both custom-built CNN and several popular pre-trained models and also leverages transfer learning to boost diagnostic accuracy and computational efficiency.

🧾 Dataset Description  
Source: Kaggle Alzheimer’s MRI Dataset 

Total Images: 30,000 2D MRI brain scans  

Diagnostic Categories:  
Non-Demented  
Very Mild Demented  
Mild Demented  
Moderate Demented  

This rich dataset provides a robust foundation for training deep learning models to identify and classify varying stages of Alzheimer’s disease.

🧠 Model Architectures  
🔁 What is Transfer Learning?  
Transfer Learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second, related task. It’s particularly useful when:  
The new task has limited labeled data.  
The initial model was trained on a large, diverse dataset (like ImageNet).  
The tasks are similar in nature (e.g., both involve image classification).

💡 Why Use Transfer Learning in Medical Imaging?  
Medical datasets, especially MRI datasets, are often small and expensive to label. Training deep learning models from scratch on such datasets can lead to overfitting and poor generalization. Transfer learning addresses this by:  
Using a pre-trained network (e.g., ResNet, VGG, Inception) that has already learned rich features from millions of images.  
Fine-tuning the model to focus on the specific features relevant to Alzheimer’s disease detection from MRI scans.  

🔨 Custom CNN (CNN50)  
A tailor-made CNN architecture consisting of multiple convolutional, pooling, and fully connected layers. This model was designed from scratch to learn from raw image data and benchmark performance against established models.

🧠 Working of Pre-trained Models  
Pre-trained CNN models used in this project include:  
ResNet18   
VGG16  
InceptionV3  
MobileNetV2  
EfficientNetB0

🔧 Implementation Steps:  
Load Pre-trained Model:  
Import the model from libraries like torchvision.models or tensorflow.keras.applications.  
Remove the top (classification) layers of the network.  

Feature Extraction Layer:  
The convolutional base of the pre-trained model acts as a feature extractor.  
It captures spatial hierarchies, edges, textures, and more from MRI scans.  

Custom Classification Head:  
Add a Flatten layer to convert 2D features into 1D.  
Add one or more Dense (Fully Connected) layers.  
Use a Dropout layer (optional) to reduce overfitting.  
End with a Sigmoid Activation in the final dense layer for binary classification or Softmax for multi-class outputs.  

Compile the Model:  
Loss Function: Binary Crossentropy (for binary classification)  
Optimizer: Adam (efficient and adaptive)  
Train & Fine-tune: Optionally unfreeze the last few layers of the base model and fine-tune them on the new dataset.

📊 Results:    
The models were trained and evaluated on the Kaggle Alzheimer’s MRI dataset across four classes:  
Non-Demented  
Very Mild Demented  
Mild Demented  
Moderate Demented  

✅ Key Metrics Used:  
Accuracy: Measures how many predictions were correct.  
Loss: Quantifies the error during training.  
Confusion Matrix: Visualizes true positives, false positives, true negatives, and false negatives.  

📈 Performance Highlights:  
Model - ResNet18  
Accuracy(%) - 98.38%


🧾 Conclusion:  
This study demonstrates that deep learning and transfer learning can effectively classify Alzheimer's disease stages using brain MRI scans. The use of pre-trained models significantly improves both the efficiency and accuracy of diagnosis, even with limited training data.

🔍 Key Takeaways:  
ResNet18 offered the best trade-off between accuracy and computation.  
Transfer learning reduces the need for large datasets and extensive training.  
Custom CNNs, while informative, underperform compared to large-scale pretrained networks.
