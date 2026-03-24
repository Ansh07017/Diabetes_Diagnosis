<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=007BFF&height=200&section=header&text=Diabetes%20Diagnosis&fontSize=70&fontColor=ffffff&animation=fadeIn&desc=Predictive%20Health%20Analytics&descAlignY=75" width="100%" />

# 🩺 Diabetes Diagnosis Predictor

<p align="center">
  <i>Early detection saves lives.<br>A machine learning web application that predicts the likelihood of diabetes based on patient diagnostic metrics.</i>
</p>

<a href="https://diabetesdiagnosis.streamlit.app/"><img src="https://img.shields.io/badge/🔴_Live_App-007BFF?style=for-the-badge&logoColor=white" alt="Live Demo"/></a>
<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>

<br>
<br>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=007BFF&center=true&vCenter=true&width=435&lines=Enter+Patient+Data.;Run+ML+Models.;Predict+Diabetes+Risk.;Early+Detection+Saves+Lives." alt="Typing SVG" />

</div>

---

## ⚡ Core Features

<table align="center" width="100%">
  <tr>
    <td align="center" width="33%">
      <h3>🧠 Dual ML Models</h3>
      <p>Powered by both <b>Random Forest</b> and <b>Logistic Regression</b> algorithms for robust, comparative predictions.</p>
    </td>
    <td align="center" width="33%">
      <h3>📊 Instant Inference</h3>
      <p>Real-time processing of patient health metrics (Glucose, BMI, Age, Insulin, etc.) to deliver immediate results.</p>
    </td>
    <td align="center" width="33%">
      <h3>🌐 Interactive UI</h3>
      <p>A clean, user-friendly web interface built entirely with Streamlit, requiring no frontend coding.</p>
    </td>
  </tr>
</table>

---

## 🛠️ The Technology Stack

<table align="center">
  <tr>
    <td align="center"><b>Web Framework</b></td>
    <td align="center"><b>Machine Learning</b></td>
    <td align="center"><b>Data Processing</b></td>
    <td align="center"><b>Deployment</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/Random_Forest-007BFF?style=flat&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/Logistic_Regression-007BFF?style=flat&logoColor=white" />
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white" />
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/Streamlit_Cloud-FF4B4B?style=flat&logo=streamlit&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" />
    </td>
  </tr>
</table>

---

## 🚀 Quick Start (Local Setup)

Want to run the prediction model on your own machine?

### 1. Clone the repository
```bash
git clone [https://github.com/Ansh07017/Diabetes_Diagnosis.git](https://github.com/Ansh07017/Diabetes_Diagnosis.git)
Step1: cd Diabetes_Diagnosis
Step2: pip install -r requirements.txt
Step3: streamlit run app.py
```
Your browser will automatically open to http://localhost:8501.

## 💡 How It Works

The application takes in critical diagnostic measurements from the user, including:

**Pregnancies:** Number of times pregnant
**Glucose:** Plasma glucose concentration
**Blood Pressure:** Diastolic blood pressure (mm Hg)
**BMI:** Body mass index
**Diabetes Pedigree Function:** Diabetes likelihood based on family history
**Age:** Age in years

These inputs are fed into the pre-trained Random Forest and Logistic Regression models, which analyze the patterns and output a binary classification indicating whether the patient is at high risk for diabetes.
