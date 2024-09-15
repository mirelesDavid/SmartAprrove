import React, { useState } from 'react';
import NavBar from '../Components/NavBar';
import InputLabel from '../Components/InputLabel'; 
import axiosInstance from '../axiosInstance'; // Importamos axiosInstance
import './LoanSystemPage.css'; 

const LoanSystemPage = () => {
  const [formData, setFormData] = useState({
    loan_amnt: '', 
    funded_amnt: '', 
    funded_amnt_inv: '', 
    term: '', 
    int_rate: '', 
    installment: '', 
    grade: '', 
    sub_grade: '', 
    emp_length: '', 
    home_ownership: '', 
    annual_inc: '', 
    verification_status: '', 
    issue_d: '', 
    purpose: '', 
    dti: '', 
    delinq_2yrs: '', 
    earliest_cr_line: '', 
    fico_range_low: '', 
    fico_range_high: '', 
    last_pymnt_d: '', 
    next_pymnt_d: ''
  });

  const [predictionResult, setPredictionResult] = useState(''); // Estado para manejar el resultado de la predicción
  const [explanation, setExplanation] = useState(''); // Estado para manejar la explicación del AI
  const [statusColor, setStatusColor] = useState('black'); // Estado para manejar el color del título

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async () => {
    try {
      console.log('Form Data:', formData);
  
      // Realizar el POST request a la API de Node.js
      const response = await axiosInstance.post('/loanapproval', formData);
  
      // Verificar la respuesta del servidor
      console.log('Response:', response);
  
      const result = response.data; // Acceder al resultado
  
      if (result.prediction && result.explanation) {
        // Establecer el resultado de la predicción en el título
        setPredictionResult(result.prediction);

        // Establecer la explicación proporcionada por OpenAI en el párrafo
        setExplanation(result.explanation);

        // Cambiar el color basado en la predicción
        if (result.prediction === 'Approved') {
          setStatusColor('green');
        } else {
          setStatusColor('red');
        }
      } else {
        console.error('Unexpected response format:', result);
        setPredictionResult('Unexpected response format');
        setExplanation('');
        setStatusColor('red');
      }
    } catch (error) {
      console.error('Error al enviar la solicitud:', error);
      setPredictionResult('Error processing your application');
      setExplanation('');
      setStatusColor('red');
    }
  };

  return (
    <div className="loan-system-container">
      <NavBar />
      <div className="loan-content">
        <h1 style={{ color: statusColor }}>
          {predictionResult || 'Welcome to the Loan Approval Model'}
        </h1>
        <p>
          {explanation || 'Using our advanced AI model, we can help determine if your loan will be approved or not. Please fill in the details below and select the applicable options to check your loan approval status.'}
        </p>
        <div className="form">
          {/* Primera fila de inputs */}
          <div className="form-row">
            <div className="input-group">
              <InputLabel htmlFor="loan_amnt" label="Loan Amount" />
              <input
                type="text"
                id="loan_amnt"
                name="loan_amnt"
                value={formData.loan_amnt}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="funded_amnt" label="Funded Amount" />
              <input
                type="text"
                id="funded_amnt"
                name="funded_amnt"
                value={formData.funded_amnt}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="funded_amnt_inv" label="Funded Amount by Investors" />
              <input
                type="text"
                id="funded_amnt_inv"
                name="funded_amnt_inv"
                value={formData.funded_amnt_inv}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="term" label="Term" />
              <input
                type="text"
                id="term"
                name="term"
                value={formData.term}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* Segunda fila de inputs */}
          <div className="form-row">
            <div className="input-group">
              <InputLabel htmlFor="int_rate" label="Interest Rate" />
              <input
                type="text"
                id="int_rate"
                name="int_rate"
                value={formData.int_rate}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="installment" label="Installment" />
              <input
                type="text"
                id="installment"
                name="installment"
                value={formData.installment}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="grade" label="Grade" />
              <input
                type="text"
                id="grade"
                name="grade"
                value={formData.grade}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="sub_grade" label="Sub Grade" />
              <input
                type="text"
                id="sub_grade"
                name="sub_grade"
                value={formData.sub_grade}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* Tercera fila de inputs */}
          <div className="form-row">
            <div className="input-group">
              <InputLabel htmlFor="emp_length" label="Employment Length (years)" />
              <input
                type="text"
                id="emp_length"
                name="emp_length"
                value={formData.emp_length}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="home_ownership" label="Home Ownership" />
              <input
                type="text"
                id="home_ownership"
                name="home_ownership"
                value={formData.home_ownership}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="annual_inc" label="Annual Income" />
              <input
                type="text"
                id="annual_inc"
                name="annual_inc"
                value={formData.annual_inc}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="verification_status" label="Verification Status" />
              <input
                type="text"
                id="verification_status"
                name="verification_status"
                value={formData.verification_status}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* Cuarta fila de inputs */}
          <div className="form-row">
            <div className="input-group">
              <InputLabel htmlFor="issue_d" label="Issue Date" />
              <input
                type="text"
                id="issue_d"
                name="issue_d"
                value={formData.issue_d}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="purpose" label="Purpose" />
              <input
                type="text"
                id="purpose"
                name="purpose"
                value={formData.purpose}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="dti" label="DTI (Debt to Income Ratio)" />
              <input
                type="text"
                id="dti"
                name="dti"
                value={formData.dti}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="delinq_2yrs" label="Delinquencies in Last 2 Years" />
              <input
                type="text"
                id="delinq_2yrs"
                name="delinq_2yrs"
                value={formData.delinq_2yrs}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* Quinta fila de inputs */}
          <div className="form-row">
            <div className="input-group">
              <InputLabel htmlFor="earliest_cr_line" label="Earliest Credit Line" />
              <input
                type="text"
                id="earliest_cr_line"
                name="earliest_cr_line"
                value={formData.earliest_cr_line}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="fico_range_low" label="FICO Range Low" />
              <input
                type="text"
                id="fico_range_low"
                name="fico_range_low"
                value={formData.fico_range_low}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="fico_range_high" label="FICO Range High" />
              <input
                type="text"
                id="fico_range_high"
                name="fico_range_high"
                value={formData.fico_range_high}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="last_pymnt_d" label="Last Payment Date" />
              <input
                type="text"
                id="last_pymnt_d"
                name="last_pymnt_d"
                value={formData.last_pymnt_d}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* Sexta fila de inputs */}
          <div className="form-row">
            <div className="input-group">
              <InputLabel htmlFor="next_pymnt_d" label="Next Payment Date" />
              <input
                type="text"
                id="next_pymnt_d"
                name="next_pymnt_d"
                value={formData.next_pymnt_d}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* Botón de envío */}
          <button className="cta-button" onClick={handleSubmit}>
            Submit Application
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoanSystemPage;
