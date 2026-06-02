from pathlib import Path
from flask import Flask, send_from_directory

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / 'web'
STATIC_DIR = BASE_DIR / 'src' / 'static'

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')


@app.route('/')
def index():
    return send_from_directory(str(WEB_DIR), 'index.html')


if __name__ == '__main__':
    app.run(debug=True, port=8050)
