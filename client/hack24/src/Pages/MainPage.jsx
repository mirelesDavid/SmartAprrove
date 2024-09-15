import React, { useRef } from 'react';
import NavBar from '../Components/NavBar';
import { useNavigate } from 'react-router-dom';
import './MainPage.css';

const MainPage = () => {
  const navigate = useNavigate();
  
  // Creamos refs para las secciones
  const poweredBySectionRef = useRef(null);
  const analyticsSectionRef = useRef(null);
  const bodyContainerRef = useRef(null); // Ref para el body-container

  // Función para hacer scroll a la sección "Powered By"
  const scrollToPoweredBy = () => {
    poweredBySectionRef.current.scrollIntoView({ behavior: 'smooth' });
  };

  // Función para hacer scroll a la primera "Analytics Section"
  const scrollToAnalyticsSection = () => {
    analyticsSectionRef.current.scrollIntoView({ behavior: 'smooth' });
  };

  // Función para hacer scroll al inicio del body-container (Home)
  const scrollToTop = () => {
    bodyContainerRef.current.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="main-container">
      <NavBar 
        scrollToPoweredBy={scrollToPoweredBy} 
        scrollToAnalyticsSection={scrollToAnalyticsSection}
        scrollToTop={scrollToTop} // Pasamos la función de scroll al inicio
      />
      <div className="content">
        <div className="body-container" ref={bodyContainerRef}>
          <div className="hero-section">
            <h1>AI-Powered: Making Smarter Loan Decisions in Seconds</h1>
            <p>
            Harnessing the power of PyTorch and advanced machine learning algorithms, our AI-driven solution empowers banks to predict loan outcomes with remarkable accuracy. With a 99.2% success rate in predicting loan approvals, our model dramatically reduces defaults and financial risks, surpassing traditional evaluation methods.
            </p>
            <div className="button-container">
              <button className="cta-button" onClick={() => navigate('/chatbot')}>Use AI Chatting</button>
              <button className="cta-button secondary-button" onClick={() => navigate('/loansystem')}>Loan Approval</button>
            </div>
          </div>
        </div>
        {/* Sección de logos */}
        <div className="logos-section" ref={poweredBySectionRef}>
          <h2>Powered By</h2>
          <div className="logos-container">
            <img src="/pytorch.png" alt="Company 1" className="company-logo company-logo1" />
            <img src="/openvino.png" alt="Company 2" className="company-logo company-logo2" />
            <img src="/openai.png" alt="Company 3" className="company-logo company-logo3" />
            <img src="/frida.png" alt="Company 4" className="company-logo company-logo4" />
          </div>
        </div>
        <div className="analytics-section" ref={analyticsSectionRef}>
          <div className="analytics-content">
            <img src="/ai.png" alt="Analytics Image" className="analytics-image" />
            <div className="analytics-text">
              <h3>Built-In Analytics To Track Your Loans</h3>
              <p>The use of AI ensures decisions are made based on objective criteria, reducing human error and bias, and enabling more reliable and consistent lending decisions.</p>
              <button className="cta-button analytics-button" onClick={() => navigate('/loansystem')}>Test Loan Approval Model</button>
            </div>
          </div>
        </div>
        <div className="analytics-section">
          <div className="analytics-content">
            <div className="analytics-text">
              <h3>SAFER FASTER SMARTER</h3>
              <p>Traditional loan approval processes are slow, requiring manual document verification, which can take days or weeks. This creates delays and inefficiencies for both banks and customers. AI-driven loan approval automates document processing and decision-making, reducing approval times and improving the customer experience.</p>
              
            </div>
            <img src="/bank.png" alt="Analytics Image" className="analytics-image1" />
          </div>
        </div>
      </div>
      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <p>© 2024 SmartApprove, All rights reserved.</p>
          <div className="footer-links">
            <a href="/privacy-policy">Privacy Policy</a>
            <a href="/terms-of-service">Terms of Service</a>
            <a href="/contact">Contact Us</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default MainPage;
