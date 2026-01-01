// Main JavaScript for Loan Eligibility Application

const API_BASE_URL = '';

// Show loading overlay
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

// Hide loading overlay
function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

// Scroll to form
function scrollToForm() {
    document.getElementById('eligibility').scrollIntoView({ behavior: 'smooth' });
}

// Format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Format percentage
function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 2
    }).format(value);
}

// Handle form submission
document.getElementById('loanForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form data
    const formData = new FormData(e.target);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });
    
    // Auto-calculate loan_percent_income if 0
    if (parseFloat(data.loan_percent_income) === 0 && parseFloat(data.person_income) > 0) {
        data.loan_percent_income = parseFloat(data.loan_amnt) / parseFloat(data.person_income);
    }
    
    showLoading();
    
    try {
        // Get prediction
        const predictionResponse = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!predictionResponse.ok) {
            throw new Error('Prediction failed');
        }
        
        const predictionResult = await predictionResponse.json();
        
        // Get SHAP explanation
        let shapExplanation = null;
        try {
            const shapResponse = await fetch(`${API_BASE_URL}/api/shap`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (shapResponse.ok) {
                shapExplanation = await shapResponse.json();
            }
        } catch (error) {
            console.error('SHAP explanation error:', error);
        }
        
        // Display results
        displayResults(predictionResult, shapExplanation, data);
        
        // Scroll to results
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    } finally {
        hideLoading();
    }
});

// Display prediction results
function displayResults(prediction, shapExplanation, formData) {
    const resultsSection = document.getElementById('results');
    const predictionDiv = document.getElementById('predictionResult');
    const shapDiv = document.getElementById('shapExplanation');
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Display prediction
    const isEligible = prediction.eligible === 1;
    const resultClass = isEligible ? 'result-eligible' : 'result-not-eligible';
    const icon = isEligible ? 'fa-check-circle' : 'fa-times-circle';
    
    predictionDiv.innerHTML = `
        <div class="result-card ${resultClass}">
            <div class="result-icon">
                <i class="fas ${icon}"></i>
            </div>
            <div class="result-status">${prediction.status}</div>
            <div class="result-probability">Confidence: ${formatPercentage(prediction.probability)}</div>
            <div class="result-message">${prediction.message}</div>
        </div>
    `;
    
    // Display SHAP explanation
    if (shapExplanation && shapExplanation.top_features) {
        let shapHTML = '<div class="shap-title"><i class="fas fa-chart-bar"></i> Key Factors Influencing Decision</div>';
        shapHTML += '<div class="shap-features">';
        
        shapExplanation.top_features.forEach(feature => {
            const impactClass = feature.impact === 'positive' ? 'positive' : 'negative';
            const impactBadgeClass = feature.impact === 'positive' ? 'impact-positive' : 'impact-negative';
            const impactIcon = feature.impact === 'positive' ? 'fa-arrow-up' : 'fa-arrow-down';
            
            shapHTML += `
                <div class="shap-feature ${impactClass}">
                    <div class="feature-name">${feature.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                    <div class="feature-impact">
                        <span class="impact-badge ${impactBadgeClass}">
                            <i class="fas ${impactIcon}"></i> ${feature.impact}
                        </span>
                        <span style="font-weight: 600; color: ${feature.impact === 'positive' ? 'var(--success-color)' : 'var(--error-color)'}">
                            ${feature.shap_value > 0 ? '+' : ''}${feature.shap_value.toFixed(4)}
                        </span>
                    </div>
                </div>
            `;
        });
        
        shapHTML += '</div>';
        shapDiv.innerHTML = shapHTML;
    } else {
        shapDiv.innerHTML = '<p>SHAP explanation not available.</p>';
    }
    
    // Store form data for PDF generation
    window.currentFormData = formData;
    window.currentPrediction = prediction;
    window.currentShapExplanation = shapExplanation;
}

// Download PDF
document.getElementById('downloadPdfBtn').addEventListener('click', async () => {
    if (!window.currentFormData) {
        alert('Please submit the form first.');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/generate-pdf`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(window.currentFormData)
        });
        
        if (!response.ok) {
            throw new Error('PDF generation failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `loan_report_${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate PDF. Please try again.');
    } finally {
        hideLoading();
    }
});

// New application button
document.getElementById('newApplicationBtn').addEventListener('click', () => {
    document.getElementById('loanForm').reset();
    document.getElementById('results').style.display = 'none';
    scrollToForm();
});

// Chatbot functionality
document.getElementById('sendChatBtn').addEventListener('click', sendChatMessage);
document.getElementById('chatInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendChatMessage();
    }
});

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addChatMessage(message, 'user');
    input.value = '';
    
    // Show typing indicator
    const typingId = addChatMessage('Typing...', 'bot', true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/chatbot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) {
            throw new Error('Chatbot error');
        }
        
        const data = await response.json();
        
        // Remove typing indicator
        document.getElementById(typingId).remove();
        
        // Add bot response
        addChatMessage(data.response, 'bot');
        
    } catch (error) {
        console.error('Chatbot error:', error);
        document.getElementById(typingId).remove();
        addChatMessage('Sorry, I encountered an error. Please try again later.', 'bot');
    }
}

function addChatMessage(message, type, isTyping = false) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageId = 'msg-' + Date.now();
    const messageClass = type === 'user' ? 'user-message' : 'bot-message';
    const icon = type === 'user' ? 'fa-user' : 'fa-robot';
    
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `chat-message ${messageClass}`;
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas ${icon}"></i>
            <p>${message}</p>
        </div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    return messageId;
}

// Smooth scroll for navigation
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// Auto-calculate loan_percent_income
document.getElementById('loan_amnt').addEventListener('input', calculateLoanPercent);
document.getElementById('person_income').addEventListener('input', calculateLoanPercent);

function calculateLoanPercent() {
    const loanAmnt = parseFloat(document.getElementById('loan_amnt').value) || 0;
    const personIncome = parseFloat(document.getElementById('person_income').value) || 0;
    

    if (personIncome > 0) {
        const percent = loanAmnt / personIncome;
        document.getElementById('loan_percent_income').value = percent.toFixed(4);
    }
}



