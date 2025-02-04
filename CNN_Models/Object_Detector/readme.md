# 📌 Custom Object Detection from Scratch 🚀

### **Project Overview**
This project is a **fully custom-built object detection model**, trained from scratch using **VGG16** as the backbone. The model detects objects in real-time by leveraging **classification & regression** for bounding boxes.

### **🔍 Key Features**
- **Custom Dataset**: Collected and labeled images using **Labelme**
- **Data Preprocessing**: Cleaned and prepared dataset for training
- **Model Architecture**: Used **VGG16** and fine-tuned layers for improved accuracy
- **Multi-Output Learning**: 
  - 🔹 **Classification**: Determines the object category  
  - 🔹 **Regression**: Generates bounding box coordinates  
- **Real-Time Inference**: Deployed on a **live webcam feed**
- **Easily Scalable**: Can be adapted for multiple object detection using **softmax activation**

### **🛠️ Tech Stack**
- `Python`
- `TensorFlow/Keras`
- `OpenCV`
- `Labelme`
- `NumPy`, `Matplotlib`

### **🚀 Future Enhancements**
- ✅ Implementing **YOLO, SSD, and Faster R-CNN**
- ✅ Enhancing dataset with **data augmentation**
- ✅ Optimizing model using **GridSearchCV/Optuna**
- ✅ Deploying on **TensorFlow.js / TFLite** for web & mobile apps

### **📌 How to Run**
```bash
# Clone the repository
git clone https://github.com/asghar-rizvi/Deep_Learning_Projects/CNN_Models/Object_Detector.git

# Install dependencies
pip install -r requirements.txt

# Run real-time detection
python real_time_detection.py
