import React from 'react';


const ChatMessage = ({ message }) => {
  const className = message.sender === 'user' ? 'chat outgoing' : 'chat incoming';
  return (
    <li className={className}>
      {message.sender === 'bot' && <img src="/newAi.png" alt="Logo AI" className="logoAi" />}
      <p>{message.text}</p>
    </li>
  );
};

export default ChatMessage;
