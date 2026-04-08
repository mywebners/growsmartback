const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

mongoose.connect("mongodb://127.0.0.1:27017/growsmart");

const User = require("./models/User");

// REGISTER
app.post("/register", async (req, res) => {
  const user = new User(req.body);
  await user.save();
  res.send("User Registered");
});

// LOGIN
app.post("/login", async (req, res) => {
  const user = await User.findOne(req.body);
  res.send({ user });
});

// FORGOT PASSWORD
app.post("/forgot", async (req, res) => {
  await User.updateOne(
    { email: req.body.email },
    { password: req.body.newPass }
  );
  res.send("Password Updated");
});

// app.listen(5000, () => console.log("Server running")); // DISABLED - Flask using port 5000
