"""
Flask Web Application for Agent Workflow Visualization

Provides a web interface for uploading JSON workflow files
and visualizing them as interactive Mermaid diagrams.
"""

import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow_parser import WorkflowParser
from mermaid_generator import MermaidGenerator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize services
parser = WorkflowParser()
generator = MermaidGenerator()

ALLOWED_EXTENSIONS = {'json'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and generate visualization"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only JSON files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Parse workflow
        try:
            workflow = parser.parse_workflow_file(filepath)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON file: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'error': f'Error parsing workflow: {str(e)}'}), 400
        
        # Generate visualization
        try:
            result = generator.generate_with_statistics(workflow)
            
            # Debug: print diagram to console
            print("Generated diagram:")
            print(result['diagram'])
            print("End diagram")
            
            # Clean up uploaded file
            os.remove(filepath)
            
            response_data = {
                'success': True,
                'workflow_name': workflow.name,
                'workflow_description': workflow.description,
                'diagram': result['diagram'],
                'legend': result['legend'],
                'statistics': result['statistics'],
                'complexity_score': result['complexity_score']
            }
            
            print("Response data keys:", list(response_data.keys()))
            print("Diagram length:", len(result['diagram']))
            
            return jsonify(response_data)
            
        except Exception as e:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error generating visualization: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/paste', methods=['POST'])
def paste_json():
    """Handle JSON paste from textarea"""
    try:
        # Get JSON from request
        json_data = request.get_json()
        if not json_data or 'workflow' not in json_data:
            return jsonify({'error': 'No workflow data provided'}), 400
        
        workflow_data = json_data['workflow']
        
        # Parse workflow
        try:
            workflow = parser.parse_workflow_dict(workflow_data)
        except Exception as e:
            return jsonify({'error': f'Error parsing workflow: {str(e)}'}), 400
        
        # Generate visualization
        try:
            result = generator.generate_with_statistics(workflow)
            
            # Debug: print diagram to console
            print("Generated diagram (paste):")
            print(result['diagram'])
            print("End diagram (paste)")
            
            response_data = {
                'success': True,
                'workflow_name': workflow.name,
                'workflow_description': workflow.description,
                'diagram': result['diagram'],
                'legend': result['legend'],
                'statistics': result['statistics'],
                'complexity_score': result['complexity_score']
            }
            
            print("Response data keys (paste):", list(response_data.keys()))
            print("Diagram length (paste):", len(result['diagram']))
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'error': f'Error generating visualization: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/examples')
def list_examples():
    """List available example workflows"""
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               'agent_orchestration', 'examples')
    
    examples = []
    if os.path.exists(examples_dir):
        for filename in os.listdir(examples_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(examples_dir, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    examples.append({
                        'filename': filename,
                        'name': data.get('name', filename),
                        'description': data.get('description', ''),
                        'steps': len(data.get('steps', []))
                    })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    
    return jsonify({'examples': examples})


@app.route('/example/<filename>')
def load_example(filename):
    """Load a specific example workflow"""
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               'agent_orchestration', 'examples')
    
    filepath = os.path.join(examples_dir, secure_filename(filename))
    
    if not os.path.exists(filepath) or not filename.endswith('.json'):
        return jsonify({'error': 'Example not found'}), 404
    
    try:
        with open(filepath, 'r') as f:
            workflow_data = json.load(f)
        
        # Parse and generate visualization
        workflow = parser.parse_workflow_dict(workflow_data)
        result = generator.generate_with_statistics(workflow)
        
        return jsonify({
            'success': True,
            'workflow_name': workflow.name,
            'workflow_description': workflow.description,
            'diagram': result['diagram'],
            'legend': result['legend'],
            'statistics': result['statistics'],
            'complexity_score': result['complexity_score'],
            'raw_json': workflow_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Error loading example: {str(e)}'}), 500


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


if __name__ == '__main__':
    print("Starting Agent Workflow Visualizer...")
    print("Access the web interface at: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)