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
            <h1>Examine the Potential of Genius's AI Chatting</h1>
            <p>
              At Genius, we believe in the power of artificial intelligence to transform the way you work and create. Our platform offers a suite of advanced AI tools designed to revolutionize your writing, coding, and media creation processes.
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
            <img src="/tensorflow.png" alt="Company 1" className="company-logo company-logo1" />
            <img src="/openvino.png" alt="Company 2" className="company-logo company-logo2" />
            <img src="/openai.png" alt="Company 3" className="company-logo company-logo3" />
            <img src="/frida.png" alt="Company 4" className="company-logo company-logo4" />
          </div>
        </div>
        <div className="analytics-section" ref={analyticsSectionRef}>
          <div className="analytics-content">
            <img src="/ai.png" alt="Analytics Image" className="analytics-image" />
            <div className="analytics-text">
              <h3>Built-In Analytics To Track Your NFTs</h3>
              <p>Use our built-in analytics dashboard to pull valuable insights and monitor the value of your Krypto portfolio over time.</p>
              <button className="cta-button analytics-button">View Our Pricing</button>
            </div>
          </div>
        </div>
        <div className="analytics-section">
          <div className="analytics-content">
            <div className="analytics-text">
              <h3>Built-In Analytics To Track Your NFTs</h3>
              <p>Use our built-in analytics dashboard to pull valuable insights and monitor the value of your Krypto portfolio over time.</p>
              <button className="cta-button analytics-button">View Our Pricing</button>
            </div>
            <img src="/bank.png" alt="Analytics Image" className="analytics-image1" />
          </div>
        </div>
      </div>
      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <p>© 2024 Genius AI. All rights reserved.</p>
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
