from flask import Flask, request, render_template, send_from_directory, Response, json
import ollama
import os

app = Flask(__name__)

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

        print(f"Received input: {user_input}")

        # Streaming response generator
        def generate_response():
            response = ollama.chat(
                model="gemma3:1b",
                messages=[{"role": "user", "content": user_input}],
                options={"num_ctx": 1024},
                stream=True  # Stream responses
            )
            for chunk in response:
                yield json.dumps({"response": chunk["message"]["content"]}) + "\n"

        return Response(generate_response(), content_type='application/json')

    except Exception as e:
        print(f"Error: {e}")
        return Response(json.dumps({"response": f"❌ Error occurred: {str(e)}"}), mimetype="application/json")

if __name__ == '__main__':
    app.run(debug=True, threaded=True)