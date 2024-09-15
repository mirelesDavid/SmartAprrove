import React, { useState } from 'react';
import NavBar from '../Components/NavBar';
import InputLabel from '../Components/InputLabel'; 
import axiosInstance from '../axiosInstance'; // Importamos axiosInstance
import './LoanSystemPage.css'; 

const LoanSystemPage = () => {
  const [formData, setFormData] = useState({
    income: '', // Ingresos mensuales
    loanAmount: '', // Cantidad del préstamo
    creditScore: '', // Puntuación de crédito
    employmentLength: '', // Años de empleo
    debtToIncomeRatio: '', // Relación deuda/ingresos
    loanPurpose: '', // Propósito del préstamo
    existingLoans: '', // Número de préstamos existentes
    collateralValue: '', // Valor del colateral
    age: '', // Edad del solicitante
    hasBankruptcy: false, // Bancarrota
    isHomeowner: false, // Propietario de vivienda
    hasCriminalRecord: false, // Antecedentes penales
  });

  const [approvalStatus, setApprovalStatus] = useState(''); // Estado para manejar el estatus
  const [statusColor, setStatusColor] = useState('black'); // Estado para manejar el color del título

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = async () => {
    try {
      // Muestra los datos que se van a enviar en la consola
      console.log('Form Data:', formData);
  
      // Realizar el POST request
      const response = await axiosInstance.post('/loanapproval', formData);
  
      // Verificar la respuesta del servidor
      console.log('Response:', response);
  
      // Verificar si la respuesta contiene los datos esperados
      const result = response.data.reply; // Accede correctamente a la respuesta
  
      // Verifica que el resultado sea un string antes de usar startsWith
      if (typeof result === 'string') {
        if (result.startsWith('Approved')) {
          setApprovalStatus('Approved');
          setStatusColor('green');
        } else {
          setApprovalStatus('Declined');
          setStatusColor('red');
        }
      } else {
        console.error('Unexpected response format:', result);
        setApprovalStatus('Unexpected response format');
        setStatusColor('red');
      }
    } catch (error) {
      console.error('Error al enviar la solicitud:', error);
      setApprovalStatus('Error processing your application');
      setStatusColor('red');
    }
  };
  

  return (
    <div className="loan-system-container">
      <NavBar />
      <div className="loan-content">
        <h1 style={{ color: statusColor }}>
          {approvalStatus || 'Welcome to the Loan Approval Model'}
        </h1>
        <p>
          Using our advanced AI model, we can help determine if your loan will be approved or not. 
          Please fill in the details below and select the applicable options to check your loan approval status.
        </p>
        <div className="form">
          {/* Primera fila de inputs */}
          <div className="form-row">
            <div className="input-group">
              <InputLabel htmlFor="income" label="Monthly Income" />
              <input
                type="text"
                id="income"
                name="income"
                value={formData.income}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="loanAmount" label="Loan Amount" />
              <input
                type="text"
                id="loanAmount"
                name="loanAmount"
                value={formData.loanAmount}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="creditScore" label="Credit Score" />
              <input
                type="text"
                id="creditScore"
                name="creditScore"
                value={formData.creditScore}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* Segunda fila de inputs */}
          <div className="form-row">
            <div className="input-group">
              <InputLabel htmlFor="employmentLength" label="Employment Length (years)" />
              <input
                type="text"
                id="employmentLength"
                name="employmentLength"
                value={formData.employmentLength}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="debtToIncomeRatio" label="Debt to Income Ratio" />
              <input
                type="text"
                id="debtToIncomeRatio"
                name="debtToIncomeRatio"
                value={formData.debtToIncomeRatio}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="loanPurpose" label="Loan Purpose" />
              <input
                type="text"
                id="loanPurpose"
                name="loanPurpose"
                value={formData.loanPurpose}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* Tercera fila de inputs */}
          <div className="form-row">
            <div className="input-group">
              <InputLabel htmlFor="existingLoans" label="Existing Loans" />
              <input
                type="text"
                id="existingLoans"
                name="existingLoans"
                value={formData.existingLoans}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="collateralValue" label="Collateral Value" />
              <input
                type="text"
                id="collateralValue"
                name="collateralValue"
                value={formData.collateralValue}
                onChange={handleChange}
              />
            </div>
            <div className="input-group">
              <InputLabel htmlFor="age" label="Age" />
              <input
                type="text"
                id="age"
                name="age"
                value={formData.age}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* Fila de checkboxes */}
          <div className="form-row checkboxes">
            <div className="checkbox-group">
              <InputLabel htmlFor="hasBankruptcy" label="Has Bankruptcy?" />
              <input
                type="checkbox"
                id="hasBankruptcy"
                name="hasBankruptcy"
                checked={formData.hasBankruptcy}
                onChange={handleChange}
              />
            </div>
            <div className="checkbox-group">
              <InputLabel htmlFor="isHomeowner" label="Is Homeowner?" />
              <input
                type="checkbox"
                id="isHomeowner"
                name="isHomeowner"
                checked={formData.isHomeowner}
                onChange={handleChange}
              />
            </div>
            <div className="checkbox-group">
              <InputLabel htmlFor="hasCriminalRecord" label="Has Criminal Record?" />
              <input
                type="checkbox"
                id="hasCriminalRecord"
                name="hasCriminalRecord"
                checked={formData.hasCriminalRecord}
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
