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
            try:
                response = ollama.chat(
                    model="mistral",
                    messages=[{"role": "user", "content": user_input}],
                    options={"num_ctx": 512},  # Reduce context for better performance
                    stream=True
                )
                for chunk in response:
                    yield f"data: {json.dumps({'response': chunk['message']['content']})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'response': f'❌ Error: {str(e)}'})}\n\n"

        return Response(generate_response(), content_type="text/event-stream")

    except Exception as e:
        print(f"Error: {e}")
        return Response(json.dumps({"response": f"❌ Error occurred: {str(e)}"}), mimetype="application/json")

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
