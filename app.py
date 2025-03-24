from flask import Flask, request, render_template, send_from_directory, Response, json, session
import requests
from dotenv import load_dotenv
import os
   
app = Flask(__name__)
load_dotenv()  # Load .env file

HUGGINGFACE_API_KEY = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
app.secret_key = os.getenv("app.secret_key")
API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"
# API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        user_input = data.get("text", "").strip()

        if not user_input:
            return Response(json.dumps({"response": "⚠️ Please enter a prompt!"}), mimetype="application/json")

        # Store conversation history before the generator function starts
        if 'chat_history' not in session:
            session['chat_history'] = []

        # Save user input
        session['chat_history'].append({"role": "user", "content": user_input})
        session.modified = True  # ✅ Ensure session updates are saved

        # Create a structured prompt for the model
        chat_prompt = "You are an AI chatbot that provides relevant and authentic responses.\n\n"


        for msg in session['chat_history']:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_prompt += f"{role}: {msg['content']}\n"
        
        chat_prompt += "Assistant:"

        print(f"Chat history sent: {chat_prompt}")

        # Call Hugging Face API outside the generator
        try:
            response = requests.post(API_URL, headers=HEADERS, json={"inputs": chat_prompt, "options": {
                             "wait_for_model": True}}, timeout=30)  # 30 sec timeout

            result = response.json()

            if "error" in result:
                session['chat_history'] = []  # Clear history if API fails
                session.modified = True
                return Response(json.dumps({"response": '❌ API Error: ' + result['error']}), mimetype="application/json")

            output_text = result[0]["generated_text"]

            # Store bot response in session before returning
            session['chat_history'].append(
                {"role": "bot", "content": output_text})
            session.modified = True  # ✅ Ensure session updates are saved

            return Response(json.dumps({"response": output_text}), mimetype="application/json")

        except Exception as e:
            return Response(json.dumps({"response": f'❌ Error: {str(e)}'}), mimetype="application/json")

    except Exception as e:
        print(f"Error: {e}")
        return Response(json.dumps({"response": f"❌ Error occurred: {str(e)}"}), mimetype="application/json")

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
