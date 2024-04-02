from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bicepcurl')
def bicepcurl():
    result = subprocess.run(['python', r'C:\Users\hp\HPE\flask_pose_counter\static\bicepcurl.py'], capture_output=True, text=True)
    return render_template('bicepcurl.html', result=result.stdout)

@app.route('/plank')
def plank():
    result = subprocess.run(['python', r'C:\Users\hp\HPE\flask_pose_counter\static\plank.py'], capture_output=True, text=True)
    return render_template('plank.html', result=result.stdout)

@app.route('/pushupcounter')
def pushupcounter():
    result = subprocess.run(['python', r'C:\Users\hp\HPE\flask_pose_counter\static\pushupcounter.py'], capture_output=True, text=True)
    return render_template('pushupcounter.html', result=result.stdout)

@app.route('/squats')
def squats():
    
    result = subprocess.run(['python', r'C:\Users\hp\HPE\flask_pose_counter\static\Squats.py'], capture_output=True, text=True)
    return render_template('squats.html', result=result.stdout)

def dispsquats():
    result = subprocess.run(['python', r'C:\Users\hp\HPE\flask_pose_counter\static\Squats.py'], capture_output=True, text=True)
    result = result.stdout
    return render_template('squats.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
