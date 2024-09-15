import fetch from 'node-fetch';
import { config } from 'dotenv';
config();

const API_KEY = process.env.API_KEY;

export const handleLoanApproval = async (req, res) => {
  try {
    // Imprimir los datos que llegan desde el frontend
    console.log('Received loan approval data:', req.body);

    // Convertir los datos del préstamo a una cadena JSON
    const loanDataMessage = JSON.stringify(req.body);

    // Crear los parámetros de la solicitud para OpenAI
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
            content: "You are an AI loan approver. Based on the information provided by the user, evaluate whether a loan should be approved or declined. If the loan is approved, start your response with 'Approved'. If the loan is denied, start your response with 'Declined'. After that, provide a brief explanation justifying the decision."
          },
          {
            role: "user",
            content: `Loan Data: ${loanDataMessage}` // Mandamos los datos como un string
          }
        ]
      })
    };

    // Hacer la solicitud a la API de OpenAI
    const response = await fetch("https://api.openai.com/v1/chat/completions", requestOptions);
    
    // Si la respuesta no es exitosa, lanzar error
    if (!response.ok) {
      console.log('Response from OpenAI failed:', response.status, response.statusText);
      throw new Error(`OpenAI API error: ${response.status} - ${response.statusText}`);
    }

    // Procesar la respuesta de OpenAI
    const data = await response.json();
    console.log('Response from OpenAI:', data);

    // Devolver la respuesta al cliente
    res.json({ reply: data.choices[0].message.content });
  } catch (error) {
    // Manejar errores y devolver una respuesta de error al cliente
    console.error("Error in loan approval handler:", error);
    res.status(500).json({ error: "Error processing loan approval" });
  }
};
