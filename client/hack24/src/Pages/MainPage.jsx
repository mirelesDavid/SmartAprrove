import React from 'react';
import NavBar from '../Components/NavBar';
import Header from '../Components/Header';
import './MainPage.css';

const MainPage = () => {
  return (
    <div className="main-container">
      <NavBar />
      <div className="content">
        <Header />
        <div className="body-container">
          <p>Contenido principal aquí.</p>
        </div>
      </div>
    </div>
  );
};

export default MainPage;
