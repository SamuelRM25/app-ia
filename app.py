from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'API is working correctly'
    })

@app.route('/echo', methods=['POST'])
def echo():
    data = request.json
    return jsonify({
        'status': 'success',
        'message': 'Echo endpoint',
        'data': data
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)