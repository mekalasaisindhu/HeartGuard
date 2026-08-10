/**
 * HeartGuard - Frontend JavaScript
 * Handles file upload, API communication, and result display
 */

// Configuration
const API_BASE_URL = (window.location.port === '5500' || window.location.port === '5501')
    ? 'http://127.0.0.1:5000'
    : window.location.origin;

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('resultsSection');
const audioPreview = document.getElementById('audioPreview');

// State
let currentFile = null;
let currentFilePath = null;
let isDarkMode = true;
let isCompareMode = false;
let baselineData = null;
let currentAnalysis = null;
let currentLanguage = 'en';

const translations = {
    en: {
        app_title: "HeartGuard",
        subtitle: "Chronic Heart Failure Detection System",
        compare: "Compare",
        upload_title: "Upload Heart Sound",
        analyze: "Analyze Recording",
        ai_insights: "AI Analysis Insights",
        insights_placeholder: "Complete a scan to see insights.",
        normal: "Normal",
        abnormal: "Abnormal",
        confidence: "Confidence",
        bpm_label: "Heart Rate (Est.)",
        download_report: "Download Report",
        verify_btn: "Verify",
        flag_btn: "Flag Error",
        history_title: "Analysis History",
        analysis_pending: "Analysis pending...",
        analysis_result: "Analysis Result",
        waveform_header: "Heart Sound Waveform",
        spectrogram_header: "Mel Spectrogram",
        model_details: "Model Predictions",
        verify_prompt: "Is this result accurate?",
        history_empty: "No past analyses found.",
        footer_main: "HeartGuard - Hybrid ML/DL System for CHF Detection",
        footer_sub: "Combining SVM and CNN for accurate heart sound analysis"
    },
    hi: {
        app_title: "हार्टगार्ड",
        subtitle: "क्रोनिक हार्ट फेल्योर डिटेक्शन सिस्टम",
        compare: "तुलना करें",
        upload_title: "हृदय की ध्वनि अपलोड करें",
        analyze: "रिकॉर्डिंग का विश्लेषण करें",
        ai_insights: "एआई विश्लेषण अंतर्दृष्टि",
        insights_placeholder: "अंतर्दृष्टि देखने के लिए स्कैन पूरा करें।",
        normal: "सामान्य",
        abnormal: "असामान्य",
        confidence: "विश्वास",
        bpm_label: "हृदय गति",
        download_report: "रिपोर्ट डाउनलोड",
        verify_btn: "पुष्टि करें",
        flag_btn: "त्रुटि बताएं",
        history_title: "विश्लेषण इतिहास",
        analysis_pending: "विश्लेषण लंबित है...",
        analysis_result: "विश्लेषण परिणाम",
        waveform_header: "हृदय ध्वनि तरंग रूप",
        spectrogram_header: "मेल स्पेक्ट्रोग्राम",
        model_details: "मॉडल भविष्यवाणियां",
        verify_prompt: "क्या यह परिणाम सटीक है?",
        history_empty: "कोई पिछला विश्लेषण नहीं मिला।",
        footer_main: "हार्टगार्ड - CHF पहचान के लिए हाइब्रिड ML/DL सिस्टम",
        footer_sub: "सटीक हृदय ध्वनि विश्लेषण के लिए SVM और CNN का संयोजन"
    },
    te: {
        app_title: "హార్ట్‌గార్డ్",
        subtitle: "క్రానిక్ హార్ట్ ఫెయిల్యూర్ డిటెక్షన్ సిస్టమ్",
        compare: "పోల్చండి",
        upload_title: "గుండె శబ్దాన్ని అప్‌లోడ్ చేయండి",
        analyze: "విశ్లేషించు",
        ai_insights: "AI అంతర్దృష్టులు",
        insights_placeholder: "వివరాల కోసం స్కాన్ చేయండి.",
        normal: "సాధారణం",
        abnormal: "అసాధారణం",
        confidence: "నమ్మకం",
        bpm_label: "గుండె వేగం",
        download_report: "రిపోర్ట్ డౌన్‌లోడ్",
        verify_btn: "ధృవీకరించు",
        flag_btn: "తప్పు అని చెప్పు",
        history_title: "చరిత్ర",
        analysis_pending: "విశ్లేషణ జరుగుతోంది...",
        analysis_result: "విశ్లేషణ ఫలితం",
        waveform_header: "గుండె ధ్వని తరంగం",
        spectrogram_header: "మెల్ స్పెక్ట్రోగ్రామ్",
        model_details: "మోడల్ అంచనాలు",
        verify_prompt: "ఈ ఫలితం ఖచ్చితమైనదా?",
        history_empty: "గత విశ్లేషణలు ఏవీ కనుగొనబడలేదు.",
        footer_main: "హార్ట్‌గార్డ్ - CHF గుర్తింపు కోసం హైబ్రిడ్ ML/DL సిస్టమ్",
        footer_sub: "ఖచ్చితమైన గుండె శబ్ద విశ్లేషణ కోసం SVM మరియు CNN కలయిక"
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setupEventListeners();
});

function setupEventListeners() {
    // Upload zone click
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);

    // Clear button
    clearBtn.addEventListener('click', clearFile);

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeAudio);

    // Theme toggle
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);

    // Compare toggle
    document.getElementById('compareToggle').addEventListener('click', toggleCompareMode);

    // Sidebar
    document.getElementById('historyToggle').addEventListener('click', openHistory);
    document.getElementById('closeHistory').addEventListener('click', closeHistory);
    document.getElementById('sidebarOverlay').addEventListener('click', closeHistory);

    // Language selector
    document.getElementById('langSelect').addEventListener('change', (e) => {
        changeLanguage(e.target.value);
    });

    // PDF Export
    document.getElementById('downloadPDF').addEventListener('click', exportToPDF);

    // Initial history load
    loadHistory();
    initLanguage();
}

function initLanguage() {
    const savedLang = localStorage.getItem('heartguard_lang') || 'en';
    document.getElementById('langSelect').value = savedLang;
    changeLanguage(savedLang);
}

function changeLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('heartguard_lang', lang);

    // Update all elements with data-i18n
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (translations[lang][key]) {
            el.textContent = translations[lang][key];
        }
    });

    // Update specific placeholders or secondary text
    document.querySelector('.logo h1').textContent = translations[lang].app_title;
    document.querySelector('.subtitle').textContent = translations[lang].subtitle;
}

function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        isDarkMode = false;
    } else {
        document.body.classList.remove('light-theme');
        isDarkMode = true;
    }
    updateThemeIcons();
}

function toggleTheme() {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle('light-theme');
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
    updateThemeIcons();
}

function updateThemeIcons() {
    const moonIcon = document.getElementById('moonIcon');
    const sunIcon = document.getElementById('sunIcon');
    if (isDarkMode) {
        moonIcon.style.display = 'none';
        sunIcon.style.display = 'block';
    } else {
        moonIcon.style.display = 'block';
        sunIcon.style.display = 'none';
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg'];
    const validExtensions = ['.wav', '.mp3', '.flac', '.ogg'];

    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(fileExtension)) {
        showError('Invalid file type. Please upload a WAV, MP3, FLAC, or OGG file.');
        return;
    }

    currentFile = file;

    // Set audio preview
    const objectUrl = URL.createObjectURL(file);
    audioPreview.src = objectUrl;
    audioPreview.load(); // Force browser to recognize the new source

    // Update UI
    fileName.textContent = file.name;
    fileInfo.style.display = 'flex';
    uploadZone.style.display = 'none';
    analyzeBtn.disabled = false;

    // Hide previous results
    resultsSection.style.display = 'none';
}

function clearFile() {
    currentFile = null;
    currentFilePath = null;
    fileInput.value = '';

    // Reset audio preview
    if (audioPreview.src) {
        URL.revokeObjectURL(audioPreview.src);
        audioPreview.src = '';
    }

    // Reset UI
    fileInfo.style.display = 'none';
    uploadZone.style.display = 'block';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
}

async function analyzeAudio() {
    if (!currentFile) {
        showError('Please select a file first.');
        return;
    }

    try {
        // Show loading
        showLoading(true);

        // Step 1: Upload file
        const uploadResult = await uploadFile(currentFile);

        // Step 2: Analyze (process + predict)
        const analysisResult = await analyzeFile(uploadResult.file_path);

        // If in compare mode and we have no baseline, make this the baseline
        if (isCompareMode && !baselineData) {
            baselineData = analysisResult;
            displayResults(analysisResult, 'baselineAnalysis');
            document.querySelector('.baseline-placeholder').style.display = 'none';
            // Reset for the next scan which will be the "current" scan
            clearFile();
        } else {
            // Normal display or current scan in comparison
            displayResults(analysisResult, 'mainAnalysis');
            currentAnalysis = analysisResult;
            saveToHistory(analysisResult);
        }

        // Hide loading
        showLoading(false);

    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'An error occurred during analysis.');
        showLoading(false);
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Upload failed');
    }

    return await response.json();
}

async function analyzeFile(filePath) {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ file_path: filePath })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Analysis failed');
    }

    return await response.json();
}

function displayResults(data, columnId = 'mainAnalysis') {
    // Show results section
    resultsSection.style.display = 'block';

    const col = document.getElementById(columnId);

    // Display prediction
    if (data.prediction) {
        const pred = data.prediction;
        const isNormal = pred.result === 'Normal';
        const prefix = columnId === 'mainAnalysis' ? '' : 'baseline';

        // Update result badge
        const resultPanel = col.querySelector('.result-panel');
        const resultBadge = resultPanel.querySelector('.result-badge');
        resultBadge.className = `result-badge ${isNormal ? 'normal' : 'abnormal'}`;

        // Update prediction text
        col.querySelector(`#${prefix}predictionText`).textContent = pred.result;
        col.querySelector(`#${prefix}confidenceText`).textContent =
            `Confidence: ${(pred.confidence * 100).toFixed(1)}%`;

        // Update model predictions
        col.querySelector(`#${prefix}svmPrediction`).textContent = pred.svm_prediction;
        col.querySelector(`#${prefix}svmConfidence`).textContent =
            `${(pred.svm_confidence * 100).toFixed(1)}%`;

        col.querySelector(`#${prefix}cnnPrediction`).textContent = pred.cnn_prediction;
        col.querySelector(`#${prefix}cnnConfidence`).textContent =
            `${(pred.cnn_confidence * 100).toFixed(1)}%`;

        // Update Gauge & BPM
        updateGauge(pred.confidence, prefix);
        if (pred.bpm) {
            animateValue(`${prefix}bpmText`, 0, Math.round(pred.bpm), 1000);
        } else {
            const bpmEl = col.querySelector(`#${prefix}bpmText`);
            if (bpmEl) bpmEl.textContent = '--';
        }

        // Update Insights
        const insightsList = col.querySelector(`#${prefix}insightsList`);
        if (insightsList && pred.insights) {
            insightsList.innerHTML = pred.insights.map(insight => {
                const isWarning = insight.includes('caution') || insight.includes('disagreement') || insight.includes('Lower confidence') || insight.includes('Tachycardia') || insight.includes('Bradycardia') || insight.includes('Identified');
                return `
                    <div class="insight-item ${isWarning ? 'warning' : 'positive'}">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="16" x2="12" y2="12"></line>
                            <line x1="12" y1="8" x2="12.01" y2="8"></line>
                        </svg>
                        <span>${insight}</span>
                    </div>
                `;
            }).join('');
        }
    }

    // Display visualizations with Plotly
    if (data.plotly_data) {
        const pd = data.plotly_data;
        const waveformId = columnId === 'mainAnalysis' ? 'waveformChart' : 'baselineWaveformChart';
        const specId = columnId === 'mainAnalysis' ? 'spectrogramChart' : 'baselineSpectrogramChart';

        // Ensure baseline IDs exist if needed
        if (columnId === 'baselineAnalysis' && !document.getElementById(waveformId)) {
            prepareBaselineColumn(col);
        }

        renderPlotlyWaveform(waveformId, pd.waveform, pd.anomalies);
        renderPlotlySpectrogram(specId, pd.spectrogram);
    }
}

function prepareBaselineColumn(col) {
    // Clone the standard visualization grid into the baseline column
    const template = document.getElementById('mainAnalysis').innerHTML;
    col.innerHTML = `
        <div class="column-header"><h3>Baseline Scan</h3></div>
        ${template.replace(/id="/g, 'id="baseline')}
    `;
}

function renderPlotlyWaveform(containerId, data, anomalies) {
    const container = document.getElementById(containerId);
    container.innerHTML = ''; // Clear loading message

    const trace = {
        x: data.x,
        y: data.y,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#00d4ff', width: 1 },
        name: 'Heart Sound'
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { t: 10, r: 10, b: 40, l: 40 },
        xaxis: {
            title: 'Time (s)',
            color: isDarkMode ? '#a0aec0' : '#475569',
            gridcolor: 'rgba(255,255,255,0.05)'
        },
        yaxis: {
            title: 'Amplitude',
            color: isDarkMode ? '#a0aec0' : '#475569',
            gridcolor: 'rgba(255,255,255,0.05)'
        },
        showlegend: false,
        height: 250
    };

    // Add anomaly markers
    if (anomalies && anomalies.length > 0) {
        const markers = {
            x: anomalies.map(a => a.time),
            y: anomalies.map(a => 0.8), // Fixed height for visibility
            mode: 'markers',
            type: 'scatter',
            marker: {
                symbol: 'pin',
                size: 12,
                color: '#ef4444'
            },
            name: 'Anomaly Detected',
            hovertext: anomalies.map(a => `Anomaly Intensity: ${a.intensity.toFixed(3)}`)
        };
        Plotly.newPlot(containerId, [trace, markers], layout, { displayModeBar: false });
    } else {
        Plotly.newPlot(containerId, [trace], layout, { displayModeBar: false });
    }
}

function renderPlotlySpectrogram(containerId, data) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    const trace = {
        z: data.z,
        x: data.x,
        y: data.y,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: false
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { t: 10, r: 10, b: 40, l: 40 },
        xaxis: {
            title: 'Time (s)',
            color: isDarkMode ? '#a0aec0' : '#475569'
        },
        yaxis: {
            title: 'Frequency (Hz)',
            color: isDarkMode ? '#a0aec0' : '#475569'
        },
        height: 300
    };

    Plotly.newPlot(containerId, [trace], layout, { displayModeBar: false });
}

function toggleCompareMode() {
    isCompareMode = !isCompareMode;
    const btn = document.getElementById('compareToggle');
    const layout = document.getElementById('analysisLayout');
    const baselineCol = document.getElementById('baselineAnalysis');

    if (isCompareMode) {
        btn.classList.add('active');
        layout.classList.replace('standard-layout', 'comparison-layout');
        baselineCol.style.display = 'block';
        document.querySelector('#mainAnalysis .column-header').style.display = 'block';
    } else {
        btn.classList.remove('active');
        layout.classList.replace('comparison-layout', 'standard-layout');
        baselineCol.style.display = 'none';
        document.querySelector('#mainAnalysis .column-header').style.display = 'none';
        baselineData = null; // Reset baseline
    }

    // Refresh Plotly charts if data exists
    if (resultsSection.style.display === 'block') {
        const containers = ['waveformChart', 'spectrogramChart'];
        containers.forEach(id => {
            if (document.getElementById(id)) Plotly.Plots.resize(id);
        });
    }
}

// History Logic
function saveToHistory(data) {
    let history = JSON.parse(localStorage.getItem('heartguard_history') || '[]');
    const item = {
        id: Date.now(),
        timestamp: new Date().toLocaleString(),
        filename: currentFile ? currentFile.name : 'Unknown',
        prediction: data.prediction.result,
        confidence: data.prediction.confidence,
        bpm: data.prediction.bpm,
        data: data // Store full data for reloading
    };
    history.unshift(item);
    // Keep only last 50
    if (history.length > 50) history = history.slice(0, 50);
    localStorage.setItem('heartguard_history', JSON.stringify(history));
    updateHistoryUI();
}

function loadHistory() {
    updateHistoryUI();
}

function updateHistoryUI() {
    const historyList = document.getElementById('historyList');
    const history = JSON.parse(localStorage.getItem('heartguard_history') || '[]');

    if (history.length === 0) {
        historyList.innerHTML = '<div class="history-empty">No past analyses found.</div>';
        return;
    }

    historyList.innerHTML = history.map(item => `
        <div class="history-item" onclick="loadFromHistory(${item.id})">
            <div class="history-item-header">
                <h4>${item.filename}</h4>
                <button class="btn-delete" title="Delete from history" onclick="event.stopPropagation(); deleteFromHistory(${item.id})">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
                    </svg>
                </button>
            </div>
            <div class="history-meta">
                <span>${item.timestamp}</span>
                <span class="status-badge ${item.prediction.toLowerCase()}">${item.prediction}</span>
            </div>
            <div class="history-meta" style="margin-top: 4px;">
                <span>${Math.round(item.confidence * 100)}% Conf.</span>
                <span>${Math.round(item.bpm)} BPM</span>
            </div>
        </div>
    `).join('');
}

function deleteFromHistory(id) {
    if (!confirm('Are you sure you want to delete this analysis from history?')) return;

    let history = JSON.parse(localStorage.getItem('heartguard_history') || '[]');
    history = history.filter(item => item.id !== id);
    localStorage.setItem('heartguard_history', JSON.stringify(history));
    updateHistoryUI();
}

function loadFromHistory(id) {
    const history = JSON.parse(localStorage.getItem('heartguard_history') || '[]');
    const item = history.find(h => h.id === id);
    if (item) {
        displayResults(item.data);
        currentAnalysis = item.data;
        closeHistory();
    }
}

function openHistory() {
    document.getElementById('historySidebar').classList.add('active');
    document.getElementById('sidebarOverlay').classList.add('active');
}

function closeHistory() {
    document.getElementById('historySidebar').classList.remove('active');
    document.getElementById('sidebarOverlay').classList.remove('active');
}

// PDF Export Logic
async function exportToPDF() {
    if (!currentAnalysis) return;

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const pred = currentAnalysis.prediction;

    // Add Branding
    doc.setFontSize(22);
    doc.setTextColor(0, 212, 255);
    doc.text('HEARTGUARD ANALYSIS REPORT', 20, 30);

    doc.setFontSize(10);
    doc.setTextColor(100, 100, 100);
    doc.text(`Generated on: ${new Date().toLocaleString()}`, 20, 40);
    doc.line(20, 45, 190, 45);

    // Results Section
    doc.setFontSize(16);
    doc.setTextColor(0, 0, 0);
    doc.text('Analysis Summary', 20, 60);

    doc.setFontSize(12);
    doc.text(`Result: ${pred.result}`, 20, 75);
    doc.text(`Confidence: ${(pred.confidence * 100).toFixed(1)}%`, 20, 85);
    doc.text(`Heart Rate: ${Math.round(pred.bpm)} BPM`, 20, 95);

    // Model Specifics
    doc.setFontSize(14);
    doc.text('Model Details', 20, 115);
    doc.setFontSize(10);
    doc.text(`SVM Prediction: ${pred.svm_prediction} (${(pred.svm_confidence * 100).toFixed(1)}%)`, 20, 125);
    doc.text(`CNN Prediction: ${pred.cnn_prediction} (${(pred.cnn_confidence * 100).toFixed(1)}%)`, 20, 132);

    // Visualizations (Snapshot Charts)
    try {
        const waveform = document.getElementById('waveformChart');
        const spec = document.getElementById('spectrogramChart');

        const waveformImg = await html2canvas(waveform);
        const specImg = await html2canvas(spec);

        doc.addPage();
        doc.text('Visualizations', 20, 20);
        doc.addImage(waveformImg.toDataURL('image/png'), 'PNG', 20, 30, 170, 60);
        doc.addImage(specImg.toDataURL('image/png'), 'PNG', 20, 100, 170, 80);
    } catch (e) {
        console.error('Snapshot failed', e);
    }

    doc.save(`HeartGuard_Report_${Date.now()}.pdf`);
}

// Feedback Logic
async function submitFeedback(isCorrect) {
    if (!currentAnalysis) return;

    const feedbackData = {
        filename: currentFile ? currentFile.name : 'Unknown',
        prediction: currentAnalysis.prediction.result,
        user_label: isCorrect ? currentAnalysis.prediction.result : (currentAnalysis.prediction.result === 'Normal' ? 'Abnormal' : 'Normal'),
        is_correct: isCorrect
    };

    try {
        const response = await fetch(`${API_BASE_URL}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(feedbackData)
        });

        if (response.ok) {
            alert('Thank you for your feedback! This data will help improve the model.');
        }
    } catch (e) {
        console.error('Feedback failed', e);
    }
}

function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

function showError(message) {
    alert(`Error: ${message}`);
}

function updateGauge(confidence, prefix = '') {
    const gaugeFill = document.getElementById(`${prefix}gaugeFill`);
    if (!gaugeFill) return;

    const circumference = 2 * Math.PI * 45; // r=45

    gaugeFill.style.strokeDasharray = circumference;
    const offset = circumference - (confidence * circumference);
    gaugeFill.style.strokeDashoffset = offset;

    animateValue(`${prefix}confidencePercent`, 0, Math.round(confidence * 100), 1000, '%');
}

function animateValue(id, start, end, duration, suffix = '') {
    const obj = document.getElementById(id);
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start) + suffix;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Health check on load
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (!data.models_loaded) {
            console.warn('Models not loaded. Please train models first.');
        }
    } catch (error) {
        console.error('Backend not available:', error);
        showError('Backend server is not running. Please start the server with: python backend/app.py');
    }
}

// Check backend on load
setTimeout(checkBackendHealth, 1000);
