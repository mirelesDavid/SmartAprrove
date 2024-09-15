import React from 'react';


const ChatMessage = ({ message }) => {
  const className = message.sender === 'user' ? 'chat outgoing' : 'chat incoming';
  return (
    <li className={className}>
      {message.sender === 'bot' && <img src="/AILogo.png" alt="Logo AI" className="logoAi" style={{width:"2%", height:"2%"
        
      }}/>}
      <p>{message.text}</p>
    </li>
  );
};

export default ChatMessage;
