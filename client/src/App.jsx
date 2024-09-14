import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './Pages/MainPage';
import ChatBot from './Pages/ChatBot';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/chatbot" element={<ChatBot />} />
      </Routes>
    </Router>
  );
};

export default App;
