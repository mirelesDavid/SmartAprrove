import React from 'react';
import { useNavigate } from 'react-router-dom';
import './NavBar.css';

const NavBar = () => {
  const navigate = useNavigate();

  return (
    <div className="navbar">
      {/* Envuelve el logo y el texto dentro de un div con el onClick */}
      <div className="navbar-logo" onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>
        <img src="/ai.png" alt="Genius Logo" className="navbar-logo-img" />
        <span className="navbar-logo-text">Genius</span>
      </div>
      <div className="nav-links">
        <button className="nav-button" onClick={() => navigate('/')}>Home</button>
        <button className="nav-button" onClick={() => navigate('/chatbot')}>About Us</button>
        <button className="nav-button" onClick={() => navigate('/features')}>Features</button>
        <button className="nav-button" onClick={() => navigate('/how-to-use')}>How To Use</button>
        <button className="nav-button" onClick={() => navigate('/pricing')}>Pricing</button>
      </div>
      <button className="nav-button try-now" onClick={() => navigate('/try-now')}>Try It Now</button>
    </div>
  );
};

export default NavBar;
