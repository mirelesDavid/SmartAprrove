import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './Pages/MainPage';
import ChatBot from './Pages/ChatBot';
import LoanSystemPage from './Pages/LoanSystemPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/chatbot" element={<ChatBot />} />
        <Route path="/loansystem" element={<LoanSystemPage />} />
      </Routes>
    </Router>
  );
};

export default App;
