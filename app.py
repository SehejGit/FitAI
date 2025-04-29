import os
import uuid
from flask import Flask, request, render_template, redirect, url_for, jsonify
from analyzer.pushup_analyzer import analyze_pushup
from analyzer.curl_analyzer import analyze_bicep_curl

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create a symlink from static folder to uploads for video playback
STATIC_UPLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(STATIC_UPLOADS, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'MOV'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        # Get the exercise type
        exercise_type = request.form.get('exercise_type', 'pushup')
        
        # Process with appropriate analyzer
        if exercise_type == 'pushup':
            # Save output to static/uploads directory for web access
            output_filename = f"analyzed_{unique_filename}"
            output_path = os.path.join('static', 'uploads', output_filename)
            result = analyze_pushup(file_path, output_path)
            
            # Get the URL for the analyzed video
            output_url = url_for('static', filename=f'uploads/{output_filename}')
            
            return render_template('results.html', 
                                  result=result, 
                                  exercise_type=exercise_type,
                                  video_url=output_url)
                                  
        elif exercise_type == 'curl':
            # Save output to static/uploads directory for web access
            output_filename = f"analyzed_{unique_filename}"
            output_path = os.path.join('static', 'uploads', output_filename)
            result = analyze_bicep_curl(file_path, output_path)
            
            # Get the URL for the analyzed video
            output_url = url_for('static', filename=f'uploads/{output_filename}')
            
            return render_template('results.html', 
                                  result=result, 
                                  exercise_type=exercise_type,
                                  video_url=output_url)
        
        else:
            return jsonify({'error': 'Invalid exercise type'}), 400
    
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)