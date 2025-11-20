from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'OK'

if __name__ == '__main__':
    print('Starting minimal Flask...')
    app.run(port=5000)
