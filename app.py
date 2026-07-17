"""
Flask API for the Resume <-> JD Fit Agent.

POST /analyze  with JSON body: {"resume_text": "...", "jd_text": "..."}
Returns the full 4-step agent output as JSON.

Run with: python app.py
Requires: OPENAI_API_KEY set in your environment.
"""
from flask import Flask, request, jsonify
from chain import run_agent

app = Flask(__name__)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    resume_text = data.get("resume_text", "")
    jd_text = data.get("jd_text", "")

    if not resume_text or not jd_text:
        return jsonify({"error": "Both resume_text and jd_text are required."}), 400

    try:
        result = run_agent(resume_text, jd_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
