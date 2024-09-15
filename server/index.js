import express from "express";
import bodyParser from "body-parser";
import cors from "cors"; 
import { handleChatRequest } from "./ChatBot.js";
import { handleLoanApproval } from "./LoanApproval.js";
import { config } from 'dotenv';

config();
const app = express();
const port = 8000;

app.use(cors({
  origin: 'http://localhost:3000' 
}));


app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static("public"));

app.post("/chat", handleChatRequest);

app.post("/loanapproval", handleLoanApproval);

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
