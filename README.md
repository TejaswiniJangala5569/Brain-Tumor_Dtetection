# Brain Tumor Detection Using CNN

## About the Project

Brain Tumor Detection Using CNN is a deep learning project developed to classify brain MRI images into four different categories using a Convolutional Neural Network (CNN).

The project provides a web-based interface where users can upload an MRI image. The trained CNN model processes the uploaded image and predicts the corresponding category.

## Objective

The main objective of this project is to develop an automated MRI image classification system using deep learning techniques.

The system is designed to:

- Process brain MRI images
- Extract important image features using a CNN
- Classify MRI images into four categories
- Provide the prediction through a simple web interface

## Proposed System

The proposed system uses a trained Convolutional Neural Network to analyze brain MRI images.

The workflow is:

MRI Image → Image Preprocessing → CNN Model → Feature Extraction → Classification → Predicted Category

## CNN Model

A Convolutional Neural Network is used because CNNs are well suited for image classification tasks.

The CNN learns visual features from MRI images through convolution and pooling operations and uses these learned features to classify the input image.

The trained model is saved as:

`brain_tumor_cnn.h5`

## Four-Category Classification

The trained model classifies the input MRI image into one of four predefined categories used during model training.

The exact category names correspond to the classes used in the project's training dataset.

## Web Application

The project uses Streamlit to provide an interactive interface.

The user can:

1. Open the application.
2. Upload a brain MRI image.
3. Submit the image for analysis.
4. The CNN model processes the image.
5. The predicted category is displayed on the screen.

## Technologies Used

- Python
- TensorFlow
- Keras
- Convolutional Neural Network (CNN)
- Streamlit
- NumPy
- Pillow

## Advantages

- Automated MRI image classification
- Easy-to-use interface
- Deep learning-based image analysis
- Quick prediction after image upload
- Can be extended with additional training data

## Applications

This type of system can be used as an educational demonstration of how deep learning and computer vision can be applied to medical image classification.

It can also serve as a foundation for further research and development in medical image analysis.

## Future Scope

- Improve the CNN model using a larger dataset
- Improve classification performance
- Add prediction confidence scores
- Add model evaluation graphs
- Experiment with transfer learning models
- Deploy the application as an online web application
- Add additional medical image analysis features

## Conclusion

The Brain Tumor Detection Using CNN project demonstrates the use of deep learning and computer vision for classifying brain MRI images.

By combining a trained CNN model with a Streamlit web application, the project provides a simple interface for uploading MRI images and obtaining a classification result.

## Disclaimer

This project is developed for educational and research purposes. The model's predictions should not be treated as a medical diagnosis or as a substitute for evaluation by a qualified healthcare professional.
