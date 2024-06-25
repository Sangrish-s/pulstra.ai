async function searchPapers() {
    const topic = document.getElementById('topic').value;
    const language = document.getElementById('language').value;

    if (topic && language) {
        const response = await fetch(`/api/search_papers?topic=${topic}&language=${language}`);
        const papers = await response.json();
        displayPapers(papers);
    }
}

function displayPapers(papers) {
    const papersDiv = document.getElementById('papers');
    papersDiv.innerHTML = '<h2>Closest Papers:</h2>';
    papers.forEach((paper, index) => {
        papersDiv.innerHTML += `<p>${index + 1}: ${paper.title}</p>`;
    });
}

async function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    const response = await fetch('/api/send_message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_input: userInput })
    });
    const chat = await response.json();
    displayChat(chat);
}

function displayChat(chat) {
    const chatHistory = document.getElementById('chatHistory');
    chatHistory.innerHTML = '';
    chat.forEach(message => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${message.user ? 'user' : 'bot'}`;
        messageDiv.innerHTML = `<p>${message.user || message.bot}</p>`;
        chatHistory.appendChild(messageDiv);
    });
}

function clearChat() {
    fetch('/api/clear_chat', { method: 'POST' });
    document.getElementById('chatHistory').innerHTML = '';
}
