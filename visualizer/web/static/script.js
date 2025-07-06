// Agent Workflow Visualizer JavaScript

class WorkflowVisualizer {
    constructor() {
        this.currentWorkflow = null;
        this.currentDiagram = null;
        this.currentLegend = null;
        this.initializeEventListeners();
        this.loadExamples();
    }

    initializeEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });

        document.querySelectorAll('.result-tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchResultTab(btn.dataset.tab));
        });

        // File upload
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // JSON paste
        document.getElementById('paste-btn').addEventListener('click', () => {
            this.handleJsonPaste();
        });

        // Result actions
        document.getElementById('show-legend-btn').addEventListener('click', () => {
            this.showLegend();
        });

        document.getElementById('download-btn').addEventListener('click', () => {
            this.downloadDiagram();
        });

        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetView();
        });

        // Modal handling
        document.querySelector('.close-btn').addEventListener('click', () => {
            this.closeModal();
        });

        document.getElementById('legend-modal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('legend-modal')) {
                this.closeModal();
            }
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }

    switchResultTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.result-tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.result-tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }

    async handleFileUpload(file) {
        if (!file.name.endsWith('.json')) {
            this.showError('Please select a JSON file');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            this.showLoading();
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
                this.hideError();
            } else {
                this.showError(result.error || 'Failed to process file');
            }
        } catch (error) {
            this.showError('Error uploading file: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async handleJsonPaste() {
        const jsonInput = document.getElementById('json-input');
        const jsonText = jsonInput.value.trim();

        if (!jsonText) {
            this.showError('Please paste JSON content');
            return;
        }

        try {
            // Validate JSON
            const workflowData = JSON.parse(jsonText);
            
            this.showLoading();
            const response = await fetch('/paste', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ workflow: workflowData })
            });

            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
                this.hideError();
            } else {
                this.showError(result.error || 'Failed to process JSON');
            }
        } catch (error) {
            if (error instanceof SyntaxError) {
                this.showError('Invalid JSON format');
            } else {
                this.showError('Error processing JSON: ' + error.message);
            }
        } finally {
            this.hideLoading();
        }
    }

    async loadExamples() {
        try {
            const response = await fetch('/examples');
            const data = await response.json();
            
            const examplesList = document.getElementById('examples-list');
            examplesList.innerHTML = '';

            if (data.examples && data.examples.length > 0) {
                data.examples.forEach(example => {
                    const card = document.createElement('div');
                    card.className = 'example-card';
                    card.innerHTML = `
                        <h4>${example.name}</h4>
                        <p>${example.description || 'No description available'}</p>
                        <div class="steps-count">${example.steps} steps</div>
                    `;
                    card.addEventListener('click', () => this.loadExample(example.filename));
                    examplesList.appendChild(card);
                });
            } else {
                examplesList.innerHTML = '<p>No examples found</p>';
            }
        } catch (error) {
            console.error('Error loading examples:', error);
            document.getElementById('examples-list').innerHTML = '<p>Error loading examples</p>';
        }
    }

    async loadExample(filename) {
        try {
            this.showLoading();
            const response = await fetch(`/example/${filename}`);
            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
                this.hideError();
            } else {
                this.showError(result.error || 'Failed to load example');
            }
        } catch (error) {
            this.showError('Error loading example: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        this.currentWorkflow = result;
        this.currentDiagram = result.diagram;
        this.currentLegend = result.legend;

        // Update workflow info
        document.getElementById('workflow-name').textContent = result.workflow_name;
        document.getElementById('workflow-description').textContent = result.workflow_description || 'No description available';

        // Render diagram
        this.renderDiagram(result.diagram);

        // Update statistics
        this.updateStatistics(result.statistics, result.complexity_score);

        // Update raw JSON
        if (result.raw_json) {
            document.getElementById('raw-json-content').textContent = JSON.stringify(result.raw_json, null, 2);
        }

        // Show results section
        document.getElementById('results-section').style.display = 'block';
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
    }

    renderDiagram(diagramText) {
        const container = document.getElementById('diagram-content');
        container.innerHTML = '<div class="loading">Rendering diagram...</div>';
        
        // Wait for Mermaid to be ready
        this.waitForMermaid(() => {
            this.doRenderDiagram(diagramText, container);
        });
    }

    waitForMermaid(callback) {
        if (window.mermaidReady && typeof mermaid !== 'undefined') {
            callback();
        } else {
            console.log('Waiting for Mermaid to be ready...');
            setTimeout(() => this.waitForMermaid(callback), 100);
        }
    }

    doRenderDiagram(diagramText, container) {
        // Create a unique ID for this diagram
        const diagramId = 'mermaid-diagram-' + Date.now();
        
        try {
            console.log('Rendering diagram with text:', diagramText);
            
            // Method 1: Try the direct div approach (most reliable)
            try {
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = `<div class="mermaid">${diagramText}</div>`;
                container.appendChild(tempDiv);
                
                // Process the mermaid element
                mermaid.init(undefined, tempDiv.querySelector('.mermaid'));
                console.log('Diagram rendered successfully using div method');
                return;
            } catch (divError) {
                console.log('Div method failed, trying render API:', divError);
                container.innerHTML = '<div class="loading">Trying alternative rendering...</div>';
            }

            // Method 2: Try the render API
            mermaid.render(diagramId, diagramText, (svgCode) => {
                console.log('Diagram rendered successfully using render API');
                container.innerHTML = svgCode;
            }, (error) => {
                console.error('Error rendering diagram with render API:', error);
                this.showDiagramError(container, diagramText, error);
            });
            
        } catch (error) {
            console.error('Error setting up diagram rendering:', error);
            this.showDiagramError(container, diagramText, error.message);
        }
    }

    showDiagramError(container, diagramText, error) {
        container.innerHTML = `
            <div class="error-message show">
                <h4>Error rendering diagram</h4>
                <p><strong>Error:</strong> ${error}</p>
                <details>
                    <summary>Raw Mermaid Code (click to expand)</summary>
                    <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px; overflow: auto; max-height: 300px;">${this.escapeHtml(diagramText)}</pre>
                </details>
                <p><em>Try refreshing the page or check the browser console for more details.</em></p>
                <button onclick="location.reload()" style="margin-top: 10px; padding: 8px 16px; background: #2196f3; color: white; border: none; border-radius: 4px; cursor: pointer;">Reload Page</button>
            </div>
        `;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    updateStatistics(stats, complexityScore) {
        document.getElementById('total-steps').textContent = stats.total_steps;
        document.getElementById('total-connections').textContent = stats.connections;
        document.getElementById('complexity-score').textContent = complexityScore;
        document.getElementById('entry-points').textContent = stats.entry_points;
        document.getElementById('exit-points').textContent = stats.exit_points;

        // Update step types
        const stepTypesList = document.getElementById('step-types-list');
        stepTypesList.innerHTML = '';

        Object.entries(stats.step_types).forEach(([type, count]) => {
            const item = document.createElement('div');
            item.className = 'step-type-item';
            item.innerHTML = `
                <span class="step-type-name">${type}</span>
                <span class="step-type-count">${count}</span>
            `;
            stepTypesList.appendChild(item);
        });
    }

    showLegend() {
        if (!this.currentLegend) {
            this.showError('No legend available');
            return;
        }

        const legendContent = document.getElementById('legend-content');
        legendContent.innerHTML = '<div class="loading">Rendering legend...</div>';
        
        // Wait for Mermaid to be ready
        this.waitForMermaid(() => {
            this.doRenderLegend(this.currentLegend, legendContent);
        });

        document.getElementById('legend-modal').style.display = 'block';
    }

    doRenderLegend(legendText, container) {
        const legendId = 'legend-diagram-' + Date.now();
        
        try {
            // Method 1: Try the direct div approach (most reliable)
            try {
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = `<div class="mermaid">${legendText}</div>`;
                container.appendChild(tempDiv);
                
                // Process the mermaid element
                mermaid.init(undefined, tempDiv.querySelector('.mermaid'));
                console.log('Legend rendered successfully using div method');
                return;
            } catch (divError) {
                console.log('Legend div method failed, trying render API:', divError);
                container.innerHTML = '<div class="loading">Trying alternative rendering...</div>';
            }

            // Method 2: Try the render API
            mermaid.render(legendId, legendText, (svgCode) => {
                container.innerHTML = svgCode;
            }, (error) => {
                console.error('Error rendering legend:', error);
                container.innerHTML = `
                    <div class="error-message show">
                        Error rendering legend: ${error}
                    </div>
                `;
            });
        } catch (error) {
            console.error('Error setting up legend rendering:', error);
            container.innerHTML = `
                <div class="error-message show">
                    Error setting up legend rendering: ${error.message}
                </div>
            `;
        }
    }

    closeModal() {
        document.getElementById('legend-modal').style.display = 'none';
    }

    downloadDiagram() {
        const svgElement = document.querySelector('#diagram-content svg');
        if (!svgElement) {
            this.showError('No diagram to download');
            return;
        }

        try {
            // Create a canvas to convert SVG to PNG
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Get SVG dimensions
            const svgData = new XMLSerializer().serializeToString(svgElement);
            const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
            const svgUrl = URL.createObjectURL(svgBlob);
            
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                
                // Download the image
                const link = document.createElement('a');
                link.download = `${this.currentWorkflow.workflow_name || 'workflow'}-diagram.png`;
                link.href = canvas.toDataURL('image/png');
                link.click();
                
                URL.revokeObjectURL(svgUrl);
            };
            img.src = svgUrl;
        } catch (error) {
            console.error('Error downloading diagram:', error);
            this.showError('Error downloading diagram: ' + error.message);
        }
    }

    resetView() {
        // Hide results section
        document.getElementById('results-section').style.display = 'none';
        
        // Clear inputs
        document.getElementById('file-input').value = '';
        document.getElementById('json-input').value = '';
        
        // Reset to first tab
        this.switchTab('upload');
        
        // Clear current data
        this.currentWorkflow = null;
        this.currentDiagram = null;
        this.currentLegend = null;
        
        // Hide any error messages
        this.hideError();
    }

    showLoading() {
        document.getElementById('loading-overlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loading-overlay').style.display = 'none';
    }

    showError(message) {
        const errorElement = document.getElementById('error-message');
        errorElement.textContent = message;
        errorElement.classList.add('show');
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        const errorElement = document.getElementById('error-message');
        errorElement.classList.remove('show');
    }
}

// Initialize the visualizer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new WorkflowVisualizer();
});