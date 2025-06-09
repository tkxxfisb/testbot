let ws;
let currentLoadingElement = null;
let lastMessageType = null;
const chatContainer = document.getElementById("chat-container");
const statusElement = document.getElementById("connection-status");
const sendBtn = document.getElementById("send-btn");
const userInput = document.getElementById("user-input");
let receivedText = null;
let receivedAudio = null;
let heartbeatInterval; // 心跳定时器

function initWebSocket() {
    ws = new WebSocket("ws://localhost:8000/ws");

    ws.onopen = () => {
        statusElement.textContent = "已连接";
        statusElement.className = "connection-status status-connected";
        sendBtn.disabled = false;
        userInput.disabled = false;

        // 启动心跳定时器，每 30 秒发送一次心跳消息
        heartbeatInterval = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'heartbeat' }));
            }
        }, 30000);

    };

    ws.onmessage = (event) => {
        if (event.data instanceof Blob) {
            receivedAudio = event.data;
        } else {
            const response = JSON.parse(event.data);
            receivedText = response.text;
        }

        if (receivedText && receivedAudio) {
            if (currentLoadingElement) {
                currentLoadingElement.remove();
                currentLoadingElement = null;
            }

            let messageGroup;
            if (lastMessageType === "bot" &&
                chatContainer.lastElementChild?.classList.contains("bot-message-group")) {
                messageGroup = chatContainer.lastElementChild;
            } else {
                messageGroup = document.createElement("div");
                messageGroup.className = "message-group bot-message-group";
                chatContainer.appendChild(messageGroup);
            }

            const messageDiv = document.createElement("div");
            messageDiv.className = "message bot-message";
            messageDiv.innerHTML = `
                <div class="avatar"></div>
                <div class="message-content">${receivedText}</div>
            `;
            messageGroup.appendChild(messageDiv);
            lastMessageType = "bot";
            scrollToBottom();

            const blob = new Blob([receivedAudio], { type: 'audio/mpeg' });
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            audio.play().catch((error) => {
                console.error("音频播放失败:", error);
            });

            receivedText = null;
            receivedAudio = null;
        }
    };

    ws.onclose = () => {
        statusElement.textContent = "连接断开";
        statusElement.className = "connection-status status-disconnected";
        sendBtn.disabled = true;
        userInput.disabled = true;

        // 清除心跳定时器
        clearInterval(heartbeatInterval);
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
    };
}

function sendMessage() {
    const message = userInput.value.trim();
    if (!message || ws?.readyState !== WebSocket.OPEN) return;

    let messageGroup;
    if (lastMessageType === "user" &&
        chatContainer.lastElementChild?.classList.contains("user-message-group")) {
        messageGroup = chatContainer.lastElementChild;
    } else {
        messageGroup = document.createElement("div");
        messageGroup.className = "message-group user-message-group";
        chatContainer.appendChild(messageGroup);
    }

    const messageDiv = document.createElement("div");
    messageDiv.className = "message user-message";
    messageDiv.innerHTML = `
        <div class="message-content">${message}</div>
        <div class="avatar"></div>
    `;
    messageGroup.appendChild(messageDiv);
    lastMessageType = "user";

    if (currentLoadingElement) {
        currentLoadingElement.remove();
    }
    currentLoadingElement = document.createElement("div");
    currentLoadingElement.className = "loading-message";
    currentLoadingElement.innerHTML = `
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
        <span style="color: #666; margin-left: 4px;">正在思考...</span>
    `;
    chatContainer.appendChild(currentLoadingElement);

    userInput.value = "";
    scrollToBottom();
    ws.send(message);
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

initWebSocket();