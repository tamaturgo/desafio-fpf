class WMSSystem {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8080/api/v1';
        this.currentStream = null;
        this.processingTasks = new Set();
        this.stats = {
            totalProcessed: 0,
            totalObjects: 0,
            totalQRCodes: 0,
            totalProcessingTime: 0
        };
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkSystemHealth();
        this.loadHistory();
        this.updateStats();
        
        // Check for processing tasks every 3 seconds
        setInterval(() => this.checkProcessingTasks(), 3000);
        
        // Auto-refresh history every 30 seconds
        setInterval(() => this.loadHistory(), 30000);
    }

    setupEventListeners() {
        // Tab switching
        document.getElementById('cameraTab').addEventListener('click', () => this.switchTab('camera'));
        document.getElementById('uploadTab').addEventListener('click', () => this.switchTab('upload'));

        // Camera controls
        document.getElementById('startCamera').addEventListener('click', () => this.startCamera());
        document.getElementById('capturePhoto').addEventListener('click', () => this.capturePhoto());
        document.getElementById('stopCamera').addEventListener('click', () => this.stopCamera());

        // File upload
        document.getElementById('selectFile').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });
        
        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        document.getElementById('processUpload').addEventListener('click', () => {
            this.processUploadedImage();
        });

        // History refresh
        document.getElementById('refreshHistory').addEventListener('click', () => this.loadHistory());

        // Health check
        document.getElementById('healthCheck').addEventListener('click', () => this.checkSystemHealth());

        // Drag and drop
        const uploadSection = document.getElementById('uploadSection');
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('border-blue-400', 'bg-blue-50');
        });
        
        uploadSection.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('border-blue-400', 'bg-blue-50');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('border-blue-400', 'bg-blue-50');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
    }

    switchTab(tab) {
        const cameraTab = document.getElementById('cameraTab');
        const uploadTab = document.getElementById('uploadTab');
        const cameraSection = document.getElementById('cameraSection');
        const uploadSection = document.getElementById('uploadSection');

        if (tab === 'camera') {
            cameraTab.classList.add('bg-white', 'text-gray-900', 'shadow-sm');
            cameraTab.classList.remove('text-gray-500');
            uploadTab.classList.remove('bg-white', 'text-gray-900', 'shadow-sm');
            uploadTab.classList.add('text-gray-500');
            
            cameraSection.classList.remove('hidden');
            uploadSection.classList.add('hidden');
        } else {
            uploadTab.classList.add('bg-white', 'text-gray-900', 'shadow-sm');
            uploadTab.classList.remove('text-gray-500');
            cameraTab.classList.remove('bg-white', 'text-gray-900', 'shadow-sm');
            cameraTab.classList.add('text-gray-500');
            
            uploadSection.classList.remove('hidden');
            cameraSection.classList.add('hidden');
        }
    }

    async startCamera() {
        try {
            this.currentStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            const video = document.getElementById('video');
            const placeholder = document.getElementById('cameraPlaceholder');
            
            video.srcObject = this.currentStream;
            video.classList.remove('hidden');
            placeholder.classList.add('hidden');
            
            document.getElementById('startCamera').classList.add('hidden');
            document.getElementById('capturePhoto').classList.remove('hidden');
            document.getElementById('stopCamera').classList.remove('hidden');
            
            this.showToast('Camera iniciada com sucesso', 'success');
        } catch (error) {
            console.error('Erro ao acessar a camera:', error);
            this.showToast('Erro ao acessar a camera. Verifique as permissões.', 'error');
        }
    }

    capturePhoto() {
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        ctx.drawImage(video, 0, 0);
        
        canvas.toBlob(async (blob) => {
            const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            await this.uploadAndProcess(file);
        }, 'image/jpeg', 0.8);
    }

    stopCamera() {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
            this.currentStream = null;
        }
        
        const video = document.getElementById('video');
        const placeholder = document.getElementById('cameraPlaceholder');
        
        video.classList.add('hidden');
        placeholder.classList.remove('hidden');
        
        document.getElementById('startCamera').classList.remove('hidden');
        document.getElementById('capturePhoto').classList.add('hidden');
        document.getElementById('stopCamera').classList.add('hidden');
        
        this.showToast('Camera desativada', 'info');
    }

    handleFileSelect(file) {
        if (!file) return;
        
        if (!file.type.startsWith('image/')) {
            this.showToast('Por favor, selecione um arquivo de imagem válido', 'error');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) {
            this.showToast('Arquivo muito grande. Máximo 10MB permitido', 'error');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('uploadPreview');
            const img = document.getElementById('previewImage');
            
            img.src = e.target.result;
            preview.classList.remove('hidden');
            preview.dataset.file = JSON.stringify({
                name: file.name,
                size: file.size,
                type: file.type
            });
        };
        reader.readAsDataURL(file);
        
        // Store file for processing
        this.selectedFile = file;
    }

    async processUploadedImage() {
        if (!this.selectedFile) {
            this.showToast('Nenhum arquivo selecionado', 'error');
            return;
        }
        
        await this.uploadAndProcess(this.selectedFile);
    }

    async uploadAndProcess(file) {
        try {
            this.showLoading('Enviando imagem...');
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`${this.apiBaseUrl}/images/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            this.hideLoading();
            this.showToast('Imagem enviada para processamento', 'success');
            
            // Add to processing tasks
            this.processingTasks.add(result.task_id);
            this.showProcessingStatus(result.task_id);
            
            // Start monitoring this task
            this.monitorTask(result.task_id);
            
        } catch (error) {
            this.hideLoading();
            console.error('Erro ao fazer upload:', error);
            this.showToast('Erro ao enviar imagem: ' + error.message, 'error');
        }
    }

    async monitorTask(taskId) {
        const maxAttempts = 60; // 3 minutes maximum
        let attempts = 0;
        
        const checkStatus = async () => {
            try {
                attempts++;
                const response = await fetch(`${this.apiBaseUrl}/results/${taskId}`);
                
                if (response.status === 202) {
                    // Still processing
                    if (attempts < maxAttempts) {
                        setTimeout(checkStatus, 3000);
                    } else {
                        this.showToast('Timeout: Processamento demorou mais que o esperado', 'warning');
                        this.hideProcessingStatus();
                        this.processingTasks.delete(taskId);
                    }
                    return;
                }
                
                if (response.ok) {
                    const result = await response.json();
                    this.hideProcessingStatus();
                    this.processingTasks.delete(taskId);
                    this.displayResult(result, taskId);
                    this.loadHistory();
                    this.updateStats();
                    this.showToast('Processamento concluído com sucesso!', 'success');
                } else {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
            } catch (error) {
                console.error('Erro ao verificar status:', error);
                if (attempts < maxAttempts) {
                    setTimeout(checkStatus, 3000);
                } else {
                    this.hideProcessingStatus();
                    this.processingTasks.delete(taskId);
                    this.showToast('Erro ao verificar status do processamento', 'error');
                }
            }
        };
        
        checkStatus();
    }

    showProcessingStatus(taskId) {
        const statusDiv = document.getElementById('processingStatus');
        const taskIdSpan = document.getElementById('processingTaskId');
        
        taskIdSpan.textContent = `Task ID: ${taskId}`;
        statusDiv.classList.remove('hidden');
    }

    hideProcessingStatus() {
        document.getElementById('processingStatus').classList.add('hidden');
    }

    displayResult(result, taskId) {
        const container = document.getElementById('currentResult');
        
        const html = `
            <div class="border rounded-lg p-4">
                <div class="flex justify-between items-start mb-4">
                    <h3 class="font-semibold text-gray-900">Resultado do Processamento</h3>
                    <span class="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">
                        ${result.status || 'COMPLETED'}
                    </span>
                </div>
                
                ${result.scan_metadata ? `
                    <div class="mb-4 p-3 bg-gray-50 rounded">
                        <h4 class="font-medium text-gray-800 mb-2">Metadados</h4>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            <div>
                                <span class="text-gray-600">Processamento:</span>
                                <span class="font-medium">${result.scan_metadata.processing_time_ms || 0}ms</span>
                            </div>
                            <div>
                                <span class="text-gray-600">Timestamp:</span>
                                <span class="font-medium">${new Date(result.scan_metadata.timestamp).toLocaleString()}</span>
                            </div>
                        </div>
                    </div>
                ` : ''}
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <h4 class="font-medium text-gray-800 mb-2 flex items-center">
                            <span class="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                            Objetos Detectados (${(result.detected_objects || []).length})
                        </h4>
                        <div class="space-y-2 max-h-32 overflow-y-auto">
                            ${(result.detected_objects || []).map(obj => `
                                <div class="flex justify-between items-center p-2 bg-blue-50 rounded">
                                    <span class="text-sm font-medium">${obj.class || 'Desconhecido'}</span>
                                    <span class="text-xs bg-blue-200 text-blue-800 px-2 py-1 rounded">
                                        ${Math.round((obj.confidence || 0) * 100)}%
                                    </span>
                                </div>
                            `).join('')}
                            ${(result.detected_objects || []).length === 0 ? 
                                '<p class="text-sm text-gray-500 italic">Nenhum objeto detectado</p>' : ''
                            }
                        </div>
                    </div>
                    
                    <div>
                        <h4 class="font-medium text-gray-800 mb-2 flex items-center">
                            <span class="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
                            QR Codes (${(result.qr_codes || []).length})
                        </h4>
                        <div class="space-y-2 max-h-32 overflow-y-auto">
                            ${(result.qr_codes || []).map(qr => `
                                <div class="p-2 bg-purple-50 rounded">
                                    <div class="text-sm font-medium">${qr.content || 'Conteúdo não disponível'}</div>
                                    <div class="text-xs text-gray-600 break-all">${qr.qr_id || 'QR Code'}</div>
                                </div>
                            `).join('')}
                            ${(result.qr_codes || []).length === 0 ? 
                                '<p class="text-sm text-gray-500 italic">Nenhum QR Code detectado</p>' : ''
                            }
                        </div>
                    </div>
                </div>
                
                <div class="mt-4 pt-3 border-t">
                    <p class="text-xs text-gray-500">
                        Task ID: ${taskId}
                    </p>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
    }

    async loadHistory() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/results?limit=10&page=1`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.displayHistory(data.tasks || []);
            
        } catch (error) {
            console.error('Erro ao carregar histórico:', error);
            document.getElementById('resultsHistory').innerHTML = `
                <div class="text-center py-4 text-red-500">
                    <p>Erro ao carregar histórico</p>
                </div>
            `;
        }
    }

    displayHistory(tasks) {
        const container = document.getElementById('resultsHistory');
        
        if (tasks.length === 0) {
            container.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <svg class="w-12 h-12 text-gray-300 mx-auto mb-4" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
                    </svg>
                    <p>Nenhum histórico disponível</p>
                </div>
            `;
            return;
        }
        
        const html = tasks.map(task => {
            return `
                <div class="border rounded-lg p-3 hover:bg-gray-50 cursor-pointer transition-colors" 
                     onclick="wmsSystem.viewTaskDetails('${task.task_info?.task_id || task.task_id}')">
                    <div class="flex justify-between items-start mb-2">
                        <span class="text-sm font-medium text-gray-900">
                            ${task.task_info?.task_id || task.task_id || 'N/A'}
                        </span>
                        <span class="text-xs px-2 py-1 rounded-full ${
                            task.status === 'COMPLETED' || task.status === 'completed' ? 'bg-green-100 text-green-800' :
                            task.status === 'failed' ? 'bg-red-100 text-red-800' :
                            'bg-yellow-100 text-yellow-800'
                        }">
                            ${task.status || 'unknown'}
                        </span>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = html;
    }

    async viewTaskDetails(taskId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/results/${taskId}`);
            if (response.ok) {
                const result = await response.json();
                this.displayResult(result, taskId);
            }
        } catch (error) {
            console.error('Erro ao carregar detalhes:', error);
            this.showToast('Erro ao carregar detalhes da task', 'error');
        }
    }

    async updateStats() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/results?limit=1000&page=1`);
            if (response.ok) {
                const data = await response.json();
                const tasks = data.tasks || [];
                
                let totalObjects = 0;
                let totalQRCodes = 0;
                let totalProcessingTime = 0;
                let completedTasks = 0;
                
                tasks.forEach(task => {
                    if (task.status === 'COMPLETED' || task.status === 'completed') {
                        completedTasks++;
                        totalObjects += (task.detected_objects || []).length;
                        totalQRCodes += (task.qr_codes || []).length;
                        
                        if (task.scan_metadata?.processing_time_ms) {
                            totalProcessingTime += task.scan_metadata.processing_time_ms;
                        }
                    }
                });
                
                const avgTime = completedTasks > 0 ? Math.round(totalProcessingTime / completedTasks) : 0;
                
                document.getElementById('totalProcessed').textContent = completedTasks;
                document.getElementById('totalObjects').textContent = totalObjects;
                document.getElementById('totalQRCodes').textContent = totalQRCodes;
                document.getElementById('avgProcessingTime').textContent = avgTime > 0 ? `${avgTime}ms` : '0ms';
            }
        } catch (error) {
            console.error('Erro ao atualizar estatísticas:', error);
        }
    }

    async checkProcessingTasks() {
        for (const taskId of this.processingTasks) {
            try {
                const response = await fetch(`${this.apiBaseUrl}/results/${taskId}`);
                
                if (response.ok) {
                    const result = await response.json();
                    this.processingTasks.delete(taskId);
                    this.hideProcessingStatus();
                    this.displayResult(result, taskId);
                    this.loadHistory();
                    this.updateStats();
                    this.showToast('Processamento concluído!', 'success');
                }
            } catch (error) {
                console.error('Erro ao verificar task:', error);
            }
        }
    }

    async checkSystemHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const health = await response.json();
            
            const button = document.getElementById('healthCheck');
            
            if (response.ok && health.status === 'healthy') {
                button.className = 'bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium';
                button.textContent = 'Sistema Online';
            } else {
                button.className = 'bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-medium';
                button.textContent = 'Sistema Offline';
            }
        } catch (error) {
            const button = document.getElementById('healthCheck');
            button.className = 'bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-medium';
            button.textContent = 'Sistema Offline';
        }
    }

    showLoading(message = 'Carregando...') {
        document.getElementById('loadingMessage').textContent = message;
        document.getElementById('loadingModal').classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loadingModal').classList.add('hidden');
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        
        const colors = {
            success: 'bg-green-500 text-white',
            error: 'bg-red-500 text-white',
            warning: 'bg-yellow-500 text-white',
            info: 'bg-blue-500 text-white'
        };
        
        toast.className = `${colors[type]} px-4 py-2 rounded-lg shadow-lg transform transition-all duration-300 translate-x-full`;
        toast.textContent = message;
        
        container.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.classList.remove('translate-x-full');
        }, 100);
        
        // Remove after 5 seconds
        setTimeout(() => {
            toast.classList.add('translate-x-full');
            setTimeout(() => {
                container.removeChild(toast);
            }, 300);
        }, 5000);
    }
}

// Initialize the system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.wmsSystem = new WMSSystem();
});
