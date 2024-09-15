import React from 'react';

const InputLabel = ({ htmlFor, label }) => {
  return <label htmlFor={htmlFor}>{label}</label>;
};

export default InputLabel;
