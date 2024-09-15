import React from 'react';
import './BotonesChatBot.css';

const Botones = ({ onButtonClick }) => {
  const buttons = [
    { icon: "fas fa-question", text: "What is your role as a chatbot in this system?" },
    { icon: "fas fa-calendar", text: "What does the 'Installment' field mean in loan approval?" },
    { icon: "fas fa-chart-line", text: "How does my 'FICO Score' impact loan approval?" }
  ];

  return (
    <div className="buttons-container">
      {buttons.map((button, index) => (
        <div
          key={index}
          className="button"
          onClick={() => onButtonClick(button.text)}
        >
          <i className={button.icon}></i>
          <p>{button.text}</p>
        </div>
      ))}
    </div>
  );
};

export default Botones;
