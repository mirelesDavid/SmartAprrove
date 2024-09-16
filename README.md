# SmartApprove

![image](https://github.com/user-attachments/assets/d8f3f3da-673c-48ce-aff1-7c7826f72478)


## Overview

**SmartApprove** is an advanced loan decision-making system designed to provide real-time loan approvals using a combination of AI-powered technologies. It integrates a chatbot backed by OpenAI's language model (GPT-4), and a trained Machine Learning model built using PyTorch, which evaluates loan applications based on multiple financial criteria.

The project is structured with a modern tech stack, utilizing a **React** frontend, a **Node.js/Express** backend for API communication, and a **Flask** application running the PyTorch-based loan prediction model. This multi-layered architecture ensures seamless user interaction while leveraging machine learning for decision-making.

![image](https://github.com/user-attachments/assets/cef91474-eb26-4311-9e4d-8f0224e49659)


## Features

1. **AI Chatbot Powered by OpenAI**: 
   - An interactive chatbot that helps users with their loan inquiries and explains the loan approval process. The chatbot is integrated using OpenAI's GPT-4, providing instant responses and clarifications about the loan decision.
   ![image](https://github.com/user-attachments/assets/70c34ced-3486-4f9f-b647-06a2a38c9ab6)

2. **Loan Approval Model (PyTorch)**:
   - A custom-built machine learning model, trained using PyTorch, evaluates the user's loan data. The model predicts the loan's status based on financial metrics such as income, credit score, loan amount, and other key parameters.
   ![image](https://github.com/user-attachments/assets/18bcf45e-967b-48f3-9cf1-893437fbfc69)

3. **Integrated Decision Making**:
   - The chatbot and loan prediction model work in harmony. The backend API communicates with both the PyTorch model (for prediction) and OpenAI (for explanation), ensuring users receive not only a prediction but also a context-based explanation of the decision.

4. **User-Friendly Interface**:
   - A responsive and intuitive **React** frontend allows users to input their loan data easily. The system provides real-time feedback about their loan status (approved or declined) along with AI-generated insights explaining why that decision was made.

## Technology Stack

### Frontend
- **React**: The user interface is built using React, allowing for a dynamic and interactive experience. The design is responsive and accessible, enabling users to easily input their loan information.
  
### Backend
- **Node.js & Express**: The backend serves as the middleman between the React frontend, OpenAI's GPT-4, and the PyTorch model. It processes API requests, handles form submissions, and connects to both the Flask service for loan prediction and OpenAI for chatbot functionality.

### Machine Learning
- **PyTorch (Loan Prediction Model)**: A custom Machine Learning model built with PyTorch, trained on historical loan data to predict loan outcomes. It uses a neural network architecture with multiple layers to evaluate the user's financial data and determine if a loan should be approved or declined. 
    - The model takes in several key features such as:
      - Loan Amount
      - Credit Score
      - Term
      - Income
      - Employment Length
      - Debt-to-Income Ratio, and more.
    - After processing the data, the model outputs whether the loan is approved and also predicts specific metrics like interest rates and installments.

- **Flask (Model API)**: The trained PyTorch model is served via a Flask API, which receives input data from the Node.js server, processes it, and returns the loan prediction. This service also handles data preprocessing, including scaling, mapping of categories, and date handling.

![image](https://github.com/user-attachments/assets/ad07380f-7e79-40fe-a6f2-03cd22ae3f58)


### Chatbot Integration
- **OpenAI GPT-3.5**: The chatbot is powered by OpenAI's language model, which provides real-time, conversational interactions. It helps users understand the loan decision process by offering explanations based on the data processed by the ML model.
  
### Deployment & Communication
- **Axios** (For API communication): Axios is used in the React frontend to send requests to the backend and Flask services, ensuring smooth communication between the different components.
  
## Project Architecture

1. **Frontend**:
   - User inputs loan data through a form.
   - The React frontend makes a POST request to the backend to evaluate the loan.

2. **Backend**:
   - The **Node.js/Express** server receives the loan data and forwards it to the Flask API running the PyTorch model.
   - Once the model provides a prediction, the backend forwards this result to the OpenAI API to generate a response explaining the decision.
   - The final result (prediction + explanation) is returned to the React frontend.

3. **Machine Learning**:
   - The **PyTorch** model is trained on historical loan data, utilizing features such as loan amount, term, income, credit score, and more.
   - The Flask API handles the prediction by preprocessing the input data (scaling, mapping, etc.) and running it through the trained model.

4. **Chatbot**:
   - The OpenAI-powered chatbot provides an explanation based on the outcome of the loan prediction, giving users a clear understanding of why their loan was approved or declined.

## How to Run the Project

1. **Backend (Node.js)**:
   - Navigate to the backend folder and install dependencies:
     ```bash
     cd server
     npm install
     ```
   - Start the Node.js server:
     ```bash
     node index.js
     ```
  
2. **Machine Learning Model (Flask & PyTorch)**:
   - Ensure you have Python installed and the required packages:
     ```bash
     cd wrapper
     pip install all required packages
     ```
   - Run the Flask API:
     ```bash
     python app.py
     ```

3. **Frontend (React)**:
   - Navigate to the frontend folder and install dependencies:
     ```bash
     cd client/hack24
     npm install
     ```
   - Start the React app:
     ```bash
     npm start
     ```

4. **Environment Variables**:
   - Ensure your environment variables (such as the OpenAI API key) are set in a `.env` file.

## Future Enhancements

- **Deploying the Application**: Set up the application in production with cloud infrastructure such as AWS or Google Cloud for scalable use.
- **Extended Data Analysis**: Expand the machine learning model to include more financial factors and further improve the accuracy of predictions.
- **AI-Driven Suggestions**: Use the chatbot to not only explain the loan decision but also suggest ways to improve the chances of loan approval in the future.

## License

This project is licensed under the Poty License. Feel free to contribute, modify, or use it for your own loan prediction models or chatbots.

---

SmartApprove leverages cutting-edge AI technology to make the loan approval process smarter, faster, and more transparent. Whether you're a developer or a financial institution, this project offers a robust framework for integrating machine learning and AI into decision-making processes.
