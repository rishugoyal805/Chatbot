# 💬 Gemma 3:1B AI Chatbot

A sleek and interactive chatbot powered by **Ollama's Gemma 3:1B model**, built with **Flask** and **JavaScript**, designed to provide real-time AI responses with a modern chat UI.

![Chatbot Screenshot](https://via.placeholder.com/800x400?text=Gemma+3:1B+Chatbot) <!-- Replace with actual screenshot -->
 
---

## 📌 Features
✅ **Real-time AI responses** using Ollama  
✅ **Modern Chat UI** with smooth animations  
✅ **User & bot avatars** for a realistic chat feel  
✅ **Dark mode** for a sleek design  
✅ **Typing indicator** when AI is generating a response  
✅ **Auto-scroll** for long conversations  
✅ **Fully responsive & mobile-friendly**  

---

## 🚀 Installation & Setup

### **1️⃣ Install Dependencies**
Ensure you have the following installed:
- **Python** (>=3.8)
- **Ollama** ([Download Here](https://ollama.com))
- **Flask** (for the backend)

```bash
pip install flask ollama
```
### **2️⃣ Start the Ollama Server**
Ensure Ollama is running:
```bash
ollama start
```
Then, download the Gemma 3:1B model:
```bash
ollama pull gemma3:1b
```
### **3️⃣ Run the Flask App**
```bash
python app.py
```
chatbot will be accessible at:
```bash
http://127.0.0.1:5000/
```
### **🖥️ Usage**
- Open your browser and go to http://127.0.0.1:5000/
- Enter your message in the chatbox
- Watch as Gemma AI responds in real-time!
---

### **🛠️ Project Structure**
```bash
/gemma-chatbot
│── /templates
│   ├── index.html    # Chatbot UI
│── /static
│   ├── favicon.ico   # Optional icon
│── app.py            # Flask Backend
│── README.md         # Documentation
│── requirements.txt  # Python dependencies

```
### **🤝 Contributing**
🚀 Pull requests are welcome!
If you want to improve this chatbot, follow these steps:

- Fork the repository
- Create a new branch (git checkout -b feature-xyz)
- Commit your changes (git commit -m "Added new feature")
- Push to your branch (git push origin feature-xyz)
- Open a Pull Request 🚀

### **📞 Contact**
- 💡 Developed by: [Rishu Goyal]
- ✉️ **Email**: rishugoyal16800@gmail.com
- 🌐 **LinkedIn**: [Rishu](https://www.linkedin.com/in/rishu0405)
- 🧑‍💻 **GitHub**: [@rishugoyal805](https://github.com/rishugoyal805)
