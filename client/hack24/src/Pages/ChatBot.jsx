import React, { useState, useEffect } from 'react';
import ChatInput from '../Components/ChatInput';
import ChatMessage from '../Components/ChatMessage';
import NavBar from '../Components/NavBar';
import Botones from '../Components/BotonesChatBot';
import './ChatBot.css';
import axiosInstance from '../axiosInstance'; 

const ChatBot = () => {
  const [messages, setMessages] = useState([
    { text: 'Hi! In what can I help you?', sender: 'bot' }
  ]);
  const [showButtons, setShowButtons] = useState(true);

  const handleSendMessage = async (message) => {
    setShowButtons(false);
    const newMessages = [...messages, { text: message, sender: 'user' }];
    setMessages(newMessages);
  
    try {
      const response = await axiosInstance.post('/chat', { message }); 
      const data = response.data;
      setMessages([...newMessages, { text: data.reply, sender: 'bot' }]);
    } catch (error) {
      setMessages([...newMessages, { text: 'Oops! Algo salió mal. Intenta de nuevo.', sender: 'bot' }]);
    }
  };

  const handleButtonClick = (message) => {
    handleSendMessage(message);
  };

  useEffect(() => {
    const chatBox = document.querySelector('.chatbox');
    chatBox.scrollTop = chatBox.scrollHeight;
  }, []);

  return (
    <>
  <NavBar style={{position:"fixed"}}/>
    <div className="main-content">
      <h1 id='titleXd'>AI Chatting</h1>
      <div className="chatbot-container">
        <div className="chatbot">
          <div className='headerXd'>
          </div>
          {showButtons && <Botones onButtonClick={handleButtonClick} />}
          <ul className="chatbox">
            {messages.map((msg, index) => (
              <ChatMessage key={index} message={msg} />
            ))}
          </ul>
          <ChatInput onSendMessage={handleSendMessage} />
        </div>
      </div>
    </div>
    </>
  );
};

export default ChatBot;
