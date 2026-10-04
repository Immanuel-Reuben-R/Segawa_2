// Update Clock on Login Screen
setInterval(() => {
    const now = new Date();
    const clock = document.getElementById('clock');
    if (clock) {
        clock.textContent = now.toLocaleTimeString('en-US', { hour12: false });
    }
}, 1000);

// Page Navigation Logic
function goToPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show target page
    const target = document.getElementById(pageId);
    if (target) {
        target.classList.add('active');
        
        // If moving to chat, update name
        if (pageId === 'page-chat') {
            const nameInput = document.getElementById('user-name');
            const displayName = document.getElementById('display-name');
            if (nameInput.value.trim() !== '') {
                displayName.textContent = nameInput.value.trim();
            }
        }
    }
}

// Chat Logic
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

async function sendMessage() {
    const inputField = document.getElementById('user-input');
    const messageText = inputField.value.trim();
    
    if (messageText === '') return;
    
    const userName = document.getElementById('display-name').textContent;
    
    // Add user message to UI
    appendMessage(messageText, 'user-message', userName);
    inputField.value = '';
    
    // Show typing indicator
    const typingId = 'typing-' + Date.now();
    appendMessage("Thinking...", 'ai-message', 'Segawa', typingId);
    
    try {
        const response = await fetch('http://127.0.0.1:5000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: messageText })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();
        
        // Append actual AI response
        if (data.response) {
            appendMessage(data.response, 'ai-message', 'Segawa');
        } else {
            appendMessage("I ran into an issue: " + (data.error || "Unknown"), 'ai-message', 'Segawa');
        }
    } catch (error) {
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();
        appendMessage("Network disconnected. Please ensure the server is running.", 'ai-message', 'Segawa');
    }
}

function appendMessage(text, className, senderName, id = null) {
    const chatMessages = document.getElementById('chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;
    if (id) messageDiv.id = id;
    
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute:'2-digit' });
    
    messageDiv.innerHTML = `
        <div class="msg-sender">${senderName} // ${time}</div>
        <div class="msg-content">${text}</div>
    `;
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Shooting Star Cursor Effect
const canvas = document.getElementById('star-canvas');
const ctx = canvas.getContext('2d');
let particles = [];
function resizeCanvas() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
window.addEventListener('resize', resizeCanvas);
resizeCanvas();
class Particle {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.size = Math.random() * 3 + 1;
        this.speedX = Math.random() * 2 - 1;
        this.speedY = Math.random() * 2 - 1;
        this.color = 'rgba(139, 92, 246, ' + Math.random() + ')'; // Neon Purple
        this.life = 1;
        this.decay = Math.random() * 0.05 + 0.02;
    }
    update() {
        this.x += this.speedX;
        this.y += this.speedY;
        this.life -= this.decay;
    }
    draw() {
        ctx.fillStyle = this.color;
        ctx.globalAlpha = Math.max(0, this.life);
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}
document.addEventListener('mousemove', (e) => {
    if(document.getElementById('page-chat') && document.getElementById('page-chat').classList.contains('active')) {
        for (let i = 0; i < 3; i++) {
            particles.push(new Particle(e.clientX, e.clientY));
        }
    }
});
function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < particles.length; i++) {
        particles[i].update();
        particles[i].draw();
        if (particles[i].life <= 0) {
            particles.splice(i, 1);
            i--;
        }
    }
    requestAnimationFrame(animateParticles);
}
animateParticles();

function selectDirective(btn, value) {
    document.querySelectorAll('.directive-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('directive-value').value = value;
}

function handleLogin(event) {
    event.preventDefault();
    const emailInput = document.getElementById('email-input');
    const emailError = document.getElementById('email-error');
    const email = emailInput.value;
    
    // Strict Regex for standard email structure
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    
    if (!emailRegex.test(email)) {
        emailError.textContent = 'Please enter a valid email address.';
        emailError.style.display = 'block';
        emailInput.style.borderColor = '#ef4444';
        
        // Add a little shake animation for error feedback
        emailInput.style.transform = 'translateX(-5px)';
        setTimeout(() => emailInput.style.transform = 'translateX(5px)', 50);
        setTimeout(() => emailInput.style.transform = 'translateX(-5px)', 100);
        setTimeout(() => emailInput.style.transform = 'translateX(0)', 150);
        return;
    }
    
    emailError.style.display = 'none';
    emailInput.style.borderColor = '';

    const loginBtn = document.getElementById('login-btn');
    loginBtn.innerHTML = 'Sending Code...';
    loginBtn.disabled = true;

    fetch('http://127.0.0.1:5000/send_verification', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email })
    }).then(res => res.json()).then(data => {
        if (data.error) {
            emailError.textContent = data.error;
            emailError.style.display = 'block';
            loginBtn.innerHTML = 'Continue →';
            loginBtn.disabled = false;
        } else {
            // Show verification modal
            document.getElementById('verify-modal').style.display = 'block';
            document.getElementById('login-form').style.display = 'none';
        }
    }).catch(err => {
        emailError.textContent = 'Network error. Is server.py running?';
        emailError.style.display = 'block';
        loginBtn.innerHTML = 'Continue →';
        loginBtn.disabled = false;
    });
}

function handleVerify() {
    const email = document.getElementById('email-input').value;
    const code = document.getElementById('verify-code-input').value;
    const errorEl = document.getElementById('verify-error');
    
    if (code.length !== 6) {
        errorEl.textContent = 'Code must be 6 digits.';
        errorEl.style.display = 'block';
        return;
    }
    
    fetch('http://127.0.0.1:5000/verify_code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email, code: code })
    }).then(res => res.json()).then(data => {
        if (data.error) {
            errorEl.textContent = data.error;
            errorEl.style.display = 'block';
        } else {
            goToPage('page-setup');
        }
    }).catch(err => {
        errorEl.textContent = 'Network error.';
        errorEl.style.display = 'block';
    });
}

