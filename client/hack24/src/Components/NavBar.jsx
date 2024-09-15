import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './NavBar.css';

const NavBar = ({ scrollToPoweredBy, scrollToTop }) => {
  const navigate = useNavigate();
  const location = useLocation(); // Usamos useLocation para obtener la ruta actual

  // Función para manejar el click en "Home"
  const handleHomeClick = () => {
    if (location.pathname === '/') {
      // Si ya estamos en la página principal, hacemos scroll hacia el inicio
      scrollToTop();
    } else {
      // Si estamos en otra página, navegamos a la página principal
      navigate('/');
    }
  };

  return (
    <div className="navbar">
      <div className="navbar-logo" onClick={handleHomeClick} style={{ cursor: 'pointer' }}>
        <img src="/ai.png" alt="Genius Logo" className="navbar-logo-img" />
        <span className="navbar-logo-text">PotyAI</span>
      </div>
      <div className="nav-links">
        <button className="nav-button" onClick={handleHomeClick}>Home</button>
        <button className="nav-button" onClick={scrollToPoweredBy}>Powered By</button>
        {/* Navegación a la página de AI Chatting */}
        <button className="nav-button" onClick={() => navigate('/chatbot')}>AI Chatting</button>
        {/* Navegación a la página de Loan Approval */}
        <button className="nav-button" onClick={() => navigate('/loansystem')}>Loan Approval</button>
        <button className="nav-button" onClick={() => navigate('/')}>About Us</button>
      </div>
    </div>
  );
};

export default NavBar;
