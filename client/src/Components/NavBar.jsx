import React from 'react';
import { useNavigate } from 'react-router-dom';
import './NavBar.css';

const NavBar = () => {
  const navigate = useNavigate();

  return (
    <div className="navbar">
      <button className="nav-button" onClick={() => navigate('/')}><i className="fa fa-user"></i></button>
      <button className="nav-button" onClick={() => navigate('/')}><i className="fa fa-home"></i></button>
      <button className="nav-button" onClick={() => navigate('/')}><i className="fa fa-gamepad"></i></button>
      <button className="nav-button" onClick={() => navigate('/chatbot')}><i className="fa fa-robot"></i></button>
    </div>
  );
}

export default NavBar;
