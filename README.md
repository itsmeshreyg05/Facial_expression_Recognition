# Facial Expression Recognition with VGG16

This project demonstrates how to recognize facial expressions in real-time using a Convolutional Neural Network (CNN) with transfer learning from the pre-trained VGG16 model. The solution includes data preparation, model training, fine-tuning, evaluation, and live webcam inference.

---

## 📂 Dataset Structure

Organize your dataset as follows:

```
face_data/
  train/
    angry/
    happy/
    sad/
    surprise/
    neutral/
    ... (one folder per expression)
```

Each subfolder should contain images for that expression class.

---

## 🚀 Features

- **Transfer Learning**: Fine-tunes VGG16 for facial expression recognition.
- **Data Augmentation**: Improves generalization with real-time image augmentation.
- **Model Evaluation**: Plots training history and sample predictions.
- **Real-Time Inference**: Detects faces and predicts expressions from your webcam.
- **Easy Customization**: Change architecture, classes, or training parameters as needed.

---

## 🛠️ Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

Install dependencies with:

```sh
pip install tensorflow opencv-python numpy matplotlib
```

---

## 📖 How to Run

1. **Prepare your dataset** in the structure above.
2. **Open** `facial_expression_recognition_vgg16.ipynb` in Jupyter or VS Code.
3. **Run all cells** to:
   - Load and preprocess data
   - Build and train the model
   - Fine-tune and evaluate
   - Visualize predictions
   - Run real-time webcam inference

---

## 🏆 Example Results

- **Training/Validation Accuracy**: Plots show model learning progress.
- **Sample Predictions**: Visualizes true vs. predicted labels for validation images.
- **Webcam Demo**: Real-time facial expression recognition with bounding boxes and labels.

---

## 🖼️ Live Demo

After training, run the last cell to start your webcam. The model will detect faces and display the predicted expression above each face.  
Press `q` to quit the webcam window.

---

## ⚙️ Customization

- Change `img_size`, `batch_size`, or data augmentation parameters for experimentation.
- Add or remove expression classes by updating your dataset folders.
- Try unfreezing more layers of VGG16 for deeper fine-tuning.

---

## 📄 License

MIT License.  
Feel free to use, modify, and share!

---

## 🙏 Acknowledgements

- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [Keras Documentation](https://keras.io/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

Happy
