import React from "react";
import NavBar from "../Components/NavBar";
import './Wallet.css';

const Wallet = () => {
    return (
        <div className="wallet-container">
            <div className="navbar-container">
                <NavBar />
            </div>
            <div className="content-container">
                <input type="text" />
                <input type="text" />
                <input type="text" />
                <input type="text" />
                <input type="text" />
                <button className="analyzeBTN" onClick={() => console.log("bum!!")}>ANALYZE</button>
            </div>
        </div>
    );
};

export default Wallet;