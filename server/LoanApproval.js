import fetch from 'node-fetch';
import axios from 'axios';
import { config } from 'dotenv';
config();

const API_KEY = process.env.API_KEY;

export const handleLoanApproval = async (req, res) => {
  try {
    // 1. Recibir los datos del frontend
    const formData = req.body;
    console.log('Received loan approval data:', formData);

    // Verificar si formData tiene el formato esperado
    if (!formData || Object.keys(formData).length === 0) {
      console.log('No form data received or empty formData.');
      return res.status(400).json({ error: "No form data provided" });
    }

    // 2. Hacer una solicitud POST a la aplicación Flask para obtener la predicción
    console.log('Sending POST request to Flask API at http://127.0.0.1:5000/predict');
    const flaskResponse = await axios.post('http://127.0.0.1:5000/predict', formData)
      .catch(error => {
        console.error('Error in Flask request:', error.response ? error.response.status : error.message);
        throw new Error('Failed to get a response from Flask API');
      });

    // Verificar si la respuesta de Flask es válida
    if (!flaskResponse || !flaskResponse.data) {
      console.log('Flask API response is empty or invalid.');
      return res.status(500).json({ error: "Flask API did not return valid data" });
    }

    const predictionResult = flaskResponse.data.prediction; // Aquí se asume que Flask envía la predicción en 'prediction'
    console.log('Prediction result from Flask:', predictionResult);

    // 3. Crear un contexto basado en la predicción para enviarlo a OpenAI
    const loanDataMessage = JSON.stringify(formData);

    // 4. Crear los parámetros de la solicitud para OpenAI con un nuevo mensaje que no empieza con Approved o Declined
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: "You are an AI loan approver. Based on the information provided by the user, evaluate whether the loan decision is correct based on the following data and provide a brief explanation."
          },
          {
            role: "user",
            content: `Loan Data: ${loanDataMessage}. The loan status predicted by the model is: ${predictionResult}. Please provide an explanation of why this decision was made.`
          }
        ]
      })
    };

    console.log('Sending POST request to OpenAI API...');
    
    // 5. Hacer la solicitud a la API de OpenAI
    const openAiResponse = await fetch("https://api.openai.com/v1/chat/completions", requestOptions)
      .catch(error => {
        console.error('Error in OpenAI request:', error);
        throw new Error('Failed to get a response from OpenAI API');
      });

    // 6. Si la respuesta no es exitosa, lanzar error
    if (!openAiResponse.ok) {
      console.log('Response from OpenAI failed:', openAiResponse.status, openAiResponse.statusText);
      throw new Error(`OpenAI API error: ${openAiResponse.status} - ${openAiResponse.statusText}`);
    }

    // 7. Procesar la respuesta de OpenAI
    const openAiData = await openAiResponse.json();
    console.log('Response from OpenAI:', openAiData);

    // 8. Combinar la predicción del modelo Flask con la explicación de OpenAI
    const responseMessage = {
      prediction: predictionResult, // El resultado del modelo de Flask (Approved/Declined)
      explanation: openAiData.choices[0].message.content // La explicación generada por OpenAI
    };

    // 9. Enviar la respuesta al frontend
    res.json(responseMessage);

  } catch (error) {
    // 10. Manejar errores y devolver una respuesta de error al cliente
    console.error("Error in loan approval handler:", error);
    res.status(500).json({ error: "Error processing loan approval" });
  }
};
