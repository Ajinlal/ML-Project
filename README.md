# 👁️ EyeCareAI Project

EyeCareAI is designed to be simple and accessible for both developers and non-technical users. Here’s a clear breakdown of its main features and what they do:

---

## 🔐 User Authentication (Login & Registration)

- Allows users to **create an account and log in securely**.
- User passwords are **encrypted and safely stored**.
- Only **logged-in users can access sensitive pages** like uploading images and viewing predictions.
- **Keeps patient data and personal uploads private and secure**.

---

## 🖼️ Image Upload & Processing

- Users can **upload retinal scan images** through the web app.
- Checks if the uploaded file is a valid image (like `.jpg`, `.png`).
- Automatically **processes the image** (resize, normalize) before sending it to the ML model.
- **Simple, beginner-friendly upload form** — no technical knowledge needed.

---

## 🧠 Machine Learning Inference (Prediction)

- Uses a **pre-trained deep learning model** to analyze uploaded images.
- Predicts the type of retinal disease (or healthy condition).
- Displays a **confidence score** (how sure the model is about its prediction).
- Provides **quick, automated feedback** for healthcare workers or remote users.

---

## 📊 Result Display

- Prediction results are shown **directly in the web app**.
- Displays:
  - **Disease type or status** (like "Glaucoma", "Normal").
  - **Confidence score** (e.g. "87% confidence").
  - (Optional) **Uploaded image preview** alongside results.

---

## 📁 Data Management (Admin Dashboard)

- Built-in **Django Admin Panel** for managing:
  - Uploaded images and predictions.
  - User accounts.
  - (Optional) Model logs or flagged cases.
- Admins can:
  - **Delete images**.
  - **Flag uncertain predictions**.
  - **Export records** for review.

---

## ⚙️ Robust, Scalable Web Application (Django)

- Built using **Django — a secure and scalable Python web framework**.
- Easily extendable with new features like:
  - Email notifications.
  - Report generation.
  - New disease detection models.
- Django handles **database management, authentication, and backend processes**.

---

## 🖥️ ML Code Integration (`src/` Folder)

**Core ML scripts:**

| 📄 File              | 📌 Purpose                                   |
|:--------------------|:---------------------------------------------|
| `eyecare_model.py`   | Defines deep learning model and loads weights.|
| `data_loader.py`     | Prepares images for the ML model.             |
| `train.py`           | Trains the ML model on your dataset.          |
| `evaluate.py`        | Measures model performance.                  |
| `predict.py`         | Makes predictions on uploaded images.         |
| `streamlit_app.py`   | (Optional) Runs a lightweight Streamlit demo. |

---

## 📊 Streamlit Demo App (Optional)

- A simple demo built with **Streamlit**.
- Allows testing the ML model **outside Django**.
- Useful for:
  - **Quick demos**.
  - **Debugging**.
  - **Rapid testing**.

---

## 📦 Project Structure Overview

**Folder & File Description:**

| 📁 Folder/File       | 📌 Purpose                                |
|:--------------------|:-------------------------------------------|
| `data/`              | Image datasets, trained models, etc.        |
| `src/`               | Machine learning scripts.                   |
| `[your_django_app]/` | Django application files (views, models).   |
| `static/` & `media/` | CSS, JavaScript, and uploaded images.        |
| `requirements.txt`   | Python dependencies.                        |
| `manage.py`          | Django management tool.                     |

---

## 📌 Summary

- ✅ Simple, secure web interface.
- ✅ Image upload and prediction in a few clicks.
- ✅ Clean, easy-to-read prediction results.
- ✅ Admin dashboard for user and data management.
- ✅ Clean project structure ideal for beginners.

---

## ✅ How to Save a `README.md` File

1. Open **VS Code**, **Notepad++**, or any text editor.
2. Copy and paste this content.
3. Save the file with the name `README.md` in your project folder.
4. Add and commit the file to your GitHub repository:
   ```bash
   git add README.md
   git commit -m "Added project README"
   git push origin main
   ```
5. Done — it will show up automatically on your GitHub repo homepage.

