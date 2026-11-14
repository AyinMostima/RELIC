// RELIC Web Interface JavaScript

// Global configuration
const RELIC = {
    config: {
        refreshInterval: 30000, // 30 seconds
        apiBaseUrl: '/api',
        debug: false
    },
    
    // State management
    state: {
        lastUpdate: null,
        systemStatus: null,
        autoRefresh: false
    },
    
    // Initialize the application
    init: function() {
        console.log('Initializing RELIC Web Interface');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Load initial data
        this.loadSystemStatus();
        
        // Start auto-refresh if enabled
        if (this.shouldAutoRefresh()) {
            this.startAutoRefresh();
        }
        
        // Add loading states
        this.setupLoadingStates();
    },
    
    // Set up global event listeners
    setupEventListeners: function() {
        // Navigation handling
        document.addEventListener('DOMContentLoaded', () => {
            this.highlightCurrentPage();
        });
        
        // Button click handlers
        document.addEventListener('click', (event) => {
            if (event.target.matches('[data-action]')) {
                this.handleActionClick(event);
            }
        });
        
        // Form submission handlers
        document.addEventListener('submit', (event) => {
            if (event.target.matches('.ajax-form')) {
                event.preventDefault();
                this.handleFormSubmit(event);
            }
        });
        
        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('auto-refresh');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (event) => {
                if (event.target.checked) {
                    this.startAutoRefresh();
                } else {
                    this.stopAutoRefresh();
                }
            });
        }
    },
    
    // Handle action button clicks
    handleActionClick: function(event) {
        const action = event.target.dataset.action;
        const target = event.target;
        
        switch (action) {
            case 'refresh-status':
                this.refreshSystemStatus(target);
                break;
            case 'run-benchmark':
                this.runBenchmark(target);
                break;
            case 'health-check':
                this.runHealthCheck(target);
                break;
            case 'toggle-details':
                this.toggleDetails(target);
                break;
            default:
                console.warn('Unknown action:', action);
        }
    },
    
    // Handle form submissions
    handleFormSubmit: function(event) {
        const form = event.target;
        const action = form.action;
        const method = form.method || 'POST';
        const formData = new FormData(form);
        
        // Convert FormData to JSON
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        this.apiRequest(action, method, data)
            .then(response => {
                this.showMessage('Operation completed successfully', 'success');
                // Refresh relevant data
                this.loadSystemStatus();
            })
            .catch(error => {
                this.showMessage(`Operation failed: ${error.message}`, 'error');
            });
    },
    
    // Load system status
    loadSystemStatus: function() {
        return this.apiRequest('/status')
            .then(data => {
                this.state.systemStatus = data;
                this.state.lastUpdate = new Date();
                this.updateStatusDisplay(data);
                return data;
            })
            .catch(error => {
                console.error('Failed to load system status:', error);
                this.showMessage('Failed to load system status', 'error');
            });
    },
    
    // Refresh system status with loading indication
    refreshSystemStatus: function(button) {
        if (!button) return;
        
        const originalText = button.textContent;
        button.textContent = 'Refreshing...';
        button.disabled = true;
        
        this.loadSystemStatus()
            .then(() => {
                this.showMessage('Status refreshed successfully', 'success');
            })
            .catch(() => {
                this.showMessage('Failed to refresh status', 'error');
            })
            .finally(() => {
                button.textContent = originalText;
                button.disabled = false;
            });
    },
    
    // Run performance benchmark
    runBenchmark: function(button) {
        if (!button) return;
        
        const originalText = button.textContent;
        button.textContent = 'Running...';
        button.disabled = true;
        
        this.apiRequest('/health/benchmark', 'POST')
            .then(data => {
                if (data.status === 'success') {
                    this.showMessage('Benchmark completed successfully', 'success');
                    this.displayBenchmarkResults(data.data);
                } else {
                    this.showMessage('Benchmark failed: ' + data.message, 'error');
                }
            })
            .catch(error => {
                this.showMessage('Benchmark failed: ' + error.message, 'error');
            })
            .finally(() => {
                button.textContent = originalText;
                button.disabled = false;
            });
    },
    
    // Run health check
    runHealthCheck: function(button) {
        if (!button) return;
        
        const originalText = button.textContent;
        button.textContent = 'Checking...';
        button.disabled = true;
        
        this.apiRequest('/health')
            .then(data => {
                this.showMessage('Health check completed', 'success');
                this.updateStatusDisplay(data);
            })
            .catch(error => {
                this.showMessage('Health check failed: ' + error.message, 'error');
            })
            .finally(() => {
                button.textContent = originalText;
                button.disabled = false;
            });
    },
    
    // Toggle detail visibility
    toggleDetails: function(button) {
        const targetId = button.dataset.target;
        const target = document.getElementById(targetId);
        
        if (target) {
            const isVisible = !target.classList.contains('hidden');
            target.classList.toggle('hidden', isVisible);
            button.textContent = isVisible ? 'Show Details' : 'Hide Details';
        }
    },
    
    // Update status display
    updateStatusDisplay: function(data) {
        // Update overall status indicator
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator && data.data && data.data.overall_status) {
            const status = data.data.overall_status.toLowerCase();
            statusIndicator.className = `status-indicator status-${status}`;
            
            const statusText = statusIndicator.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = data.data.overall_status;
            }
        }
        
        // Update subsystem status
        const subsystemItems = document.querySelectorAll('.subsystem-item');
        if (data.data && data.data.subsystems) {
            subsystemItems.forEach(item => {
                const name = item.querySelector('.subsystem-name').textContent.toLowerCase();
                const statusElement = item.querySelector('.subsystem-status');
                
                if (data.data.subsystems[name] && statusElement) {
                    const status = data.data.subsystems[name].status || 'unknown';
                    statusElement.className = `subsystem-status status-${status.toLowerCase()}`;
                    statusElement.textContent = status.toUpperCase();
                }
            });
        }
        
        // Update last refresh time
        const lastUpdateElement = document.getElementById('last-update');
        if (lastUpdateElement) {
            lastUpdateElement.textContent = new Date().toLocaleTimeString();
        }
    },
    
    // Display benchmark results
    displayBenchmarkResults: function(results) {
        // Create or update benchmark results display
        let resultsContainer = document.getElementById('benchmark-results');
        if (!resultsContainer) {
            resultsContainer = document.createElement('div');
            resultsContainer.id = 'benchmark-results';
            resultsContainer.className = 'dashboard-card';
            
            // Find a good place to insert it
            const quickActions = document.querySelector('.quick-actions');
            if (quickActions) {
                quickActions.parentNode.insertBefore(resultsContainer, quickActions.nextSibling);
            }
        }
        
        // Build results HTML
        let html = '<h3>Benchmark Results</h3>';
        
        if (results.benchmark_results) {
            const bench = results.benchmark_results;
            
            if (bench.cpu_benchmark) {
                html += `<div class="benchmark-section">
                    <h4>CPU Performance</h4>
                    <p>Matrix multiplication: ${bench.cpu_benchmark.matrix_multiply_time.toFixed(3)}ms</p>
                    <p>Throughput: ${bench.cpu_benchmark.operations_per_second.toFixed(0)} ops/sec</p>
                </div>`;
            }
            
            if (bench.gpu_benchmark && !bench.gpu_benchmark.error) {
                html += `<div class="benchmark-section">
                    <h4>GPU Performance</h4>
                    <p>Matrix multiplication: ${bench.gpu_benchmark.matrix_multiply_time.toFixed(3)}ms</p>
                    <p>Speedup over CPU: ${bench.gpu_benchmark.speedup_over_cpu.toFixed(1)}x</p>
                    <p>Throughput: ${bench.gpu_benchmark.operations_per_second.toFixed(0)} ops/sec</p>
                </div>`;
            }
            
            if (bench.memory_info) {
                html += `<div class="benchmark-section">
                    <h4>Memory Information</h4>
                    <p>Total: ${bench.memory_info.total_gb.toFixed(1)} GB</p>
                    <p>Available: ${bench.memory_info.available_gb.toFixed(1)} GB</p>
                    <p>Usage: ${bench.memory_info.usage_percent.toFixed(1)}%</p>
                </div>`;
            }
        }
        
        html += `<p class="benchmark-timestamp">Completed: ${new Date().toLocaleString()}</p>`;
        resultsContainer.innerHTML = html;
        resultsContainer.classList.add('fade-in');
    },
    
    // Auto-refresh functionality
    startAutoRefresh: function() {
        if (this.state.refreshTimer) {
            clearInterval(this.state.refreshTimer);
        }
        
        this.state.autoRefresh = true;
        this.state.refreshTimer = setInterval(() => {
            this.loadSystemStatus();
        }, this.config.refreshInterval);
        
        console.log('Auto-refresh started');
    },
    
    stopAutoRefresh: function() {
        if (this.state.refreshTimer) {
            clearInterval(this.state.refreshTimer);
            this.state.refreshTimer = null;
        }
        
        this.state.autoRefresh = false;
        console.log('Auto-refresh stopped');
    },
    
    shouldAutoRefresh: function() {
        // Auto-refresh on health and home pages
        return window.location.pathname === '/' || 
               window.location.pathname === '/health';
    },
    
    // API request helper
    apiRequest: function(endpoint, method = 'GET', data = null) {
        const url = this.config.apiBaseUrl + endpoint;
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (data && method !== 'GET') {
            options.body = JSON.stringify(data);
        }
        
        return fetch(url, options)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            });
    },
    
    // Show user messages
    showMessage: function(message, type = 'info', duration = 3000) {
        // Remove existing messages
        const existingMessages = document.querySelectorAll('.message-toast');
        existingMessages.forEach(msg => msg.remove());
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${type} message-toast`;
        messageDiv.textContent = message;
        messageDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            max-width: 400px;
            padding: 1rem;
            border-radius: 4px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(messageDiv);
        
        // Auto-remove after duration
        setTimeout(() => {
            messageDiv.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => {
                if (messageDiv.parentNode) {
                    messageDiv.remove();
                }
            }, 300);
        }, duration);
    },
    
    // Highlight current page in navigation
    highlightCurrentPage: function() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
                link.style.borderRadius = '4px';
            }
        });
    },
    
    // Setup loading states for buttons and forms
    setupLoadingStates: function() {
        // Add loading CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            @keyframes slideOutRight {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
            
            .benchmark-section {
                margin: 1rem 0;
                padding: 1rem;
                background: #f8f9fa;
                border-radius: 4px;
                border-left: 4px solid #3498db;
            }
            
            .benchmark-section h4 {
                margin-bottom: 0.5rem;
                color: #2c3e50;
            }
            
            .benchmark-timestamp {
                margin-top: 1rem;
                font-size: 0.9rem;
                color: #666;
                font-style: italic;
            }
        `;
        document.head.appendChild(style);
    },
    
    // Utility functions
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    formatUptime: function(seconds) {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        
        if (days > 0) {
            return `${days}d ${hours}h ${minutes}m`;
        } else if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${minutes}m`;
        }
    },
    
    // Debug helper
    debug: function(...args) {
        if (this.config.debug) {
            console.log('[RELIC Debug]', ...args);
        }
    }
};

// Feature detection and polyfills
if (!window.fetch) {
    console.warn('Fetch API not supported. Some features may not work.');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => RELIC.init());
} else {
    RELIC.init();
}

// Export for use in other scripts
window.RELIC = RELIC;