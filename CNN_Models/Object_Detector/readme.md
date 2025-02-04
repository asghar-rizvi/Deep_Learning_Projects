# **Custom Object Detection Model – Built from Scratch 🚀**

**📌 Trained & Deployed on Google Colab | Real-Time Detection with OpenCV**

## **Overview**
This project demonstrates a **custom-built object detection model** using **VGG16** for feature extraction, trained entirely on **Google Colab** without a GPU. The model can detect objects in real time using **OpenCV** and is structured to be easily adaptable for different objects.

## **Key Features**
✅ **Data Collection & Labeling:** Captured images & annotated using Labelme  
✅ **Preprocessing & Training:** Data cleaning, augmentation & training on Google Colab  
✅ **Model Architecture:** Fine-tuned **VGG16** for object detection  
✅ **Multi-Output Learning:**  
   - **Regression:** Predicts bounding boxes  
   - **Classification:** Identifies the object  
✅ **Real-Time Detection:** Live webcam-based detection using **OpenCV**  
✅ **Scalable:** Can be modified to detect multiple objects with **softmax activation**  
✅ **Plug & Play:** Notebook designed for easy reusability with custom datasets  

---

## **Project Structure**

📂 **Custom_Object_Detection/** _(Main Directory)_  
 ┣ 📜 **real_time_detection.py** – Uses **OpenCV** to perform real-time detection with the trained model  
 ┣ 📜 **capture_images.ipynb** – Captures images using **OpenCV** for dataset creation  
 ┣ 📜 **object_detection.ipynb** – Complete training pipeline in **Google Colab**  
 ┣ 📂 **data/images** – Contains images 
         📂 **data/labels** annotations  
 ┣ 📜 **model.h5** – Stores the trained **.h5 model**  

---

## **Installation & Setup**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/Custom_Object_Detection.git
cd Custom_Object_Detection
```

### **2️⃣ Install Dependencies**
Ensure you have Python and the required libraries installed:  
```bash
pip install opencv-python tensorflow numpy matplotlib
```

### **3️⃣ Running Real-Time Detection**
Run the following command to use the trained model for real-time detection:  
```bash
python real_time_detection.py
```

---

## **Training the Model on Google Colab**
If you want to train the model from scratch, open `object_detection.ipynb` in **Google Colab** and follow the steps. The notebook is structured to allow **anyone to train their own model** with minimal modifications.

---

## **Future Enhancements**
🔹 Implementing **YOLO, SSD, or Faster R-CNN** for improved accuracy  
🔹 Expanding to multi-object detection with **multiple classes**  
🔹 Improving real-time inference speed  

---

## **Acknowledgments**
💡 Inspired by **@nicholasrenotte**'s work, with **custom modifications** and enhancements.  

---

### **📌 Connect & Follow**
📢 **LinkedIn:** www.linkedin.com/in/asghar-qamber-rizvi-2ba8472b5

🚀 **Try it out & let me know your thoughts!**
