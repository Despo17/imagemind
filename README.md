# 🧠 ImageMind

**ImageMind** is an intelligent image classification web application built using **TensorFlow**, **Keras**, and **Streamlit**. It uses **MobileNetV2** (via transfer learning) to classify images with high accuracy and speed. The app offers a simple UI where users can upload an image and instantly get predictions.

---

## 🚀 Features

- 📸 Upload and classify any image in real time  
- 🧠 Uses pre-trained **MobileNetV2** for feature extraction  
- ⚡ Fast and lightweight model optimized for CPU  
- 🖥️ Clean, responsive UI built with Streamlit  
- 🔄 Easily extendable to new datasets or use-cases  

---

## 🖼️ Demo Screenshot

> Replace with your actual image after uploading it to GitHub inside `assets/screenshots/`

![App Screenshot](assets/screenshots/app_screenshot.png)

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Despo17/ImageMind.git
   cd ImageMind

   Create a virtual environment and activate it:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # On Windows
Install the dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run streamlit_app.py
🧠 Model Info
Architecture: MobileNetV2

Type: Transfer Learning-based CNN

Framework: TensorFlow / Keras

Model File: visionnet_tl.h5

Dataset Used: CIFAR-10 (or mention your custom dataset)

Accuracy Achieved: ~72% (or update as needed)

📁 Project Structure
bash
Copy
Edit
ImageMind/
├── streamlit_app.py         # Main Streamlit app
├── visionnet_tl.h5          # Pretrained image classification model
├── requirements.txt         # Required packages
├── README.md                # This file
└── assets/
    └── screenshots/
        └── app_screenshot.png
🚀 Deployment Ideas
✅ [ ] Deploy on Streamlit Cloud

✅ [ ] Deploy on Hugging Face Spaces

✅ [ ] Add Dockerfile for container deployment

✅ [ ] Add multi-class support / confidence scores

👨‍💻 Author
Abilash S.

GitHub: Despo17

Profession: Final year B.Tech (CSE) | Full-Stack Developer & Data Enthusiast

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

