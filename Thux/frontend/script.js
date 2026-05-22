const chatPanel = document.getElementById("chatPanel");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const plusButton = document.getElementById("plusButton");
const plusMenu = document.getElementById("plusMenu");

let isSending = false;

plusButton.addEventListener("click", () => {
    plusMenu.classList.toggle("open");
});

document.addEventListener("click", (event) => {
    const clickedInsideMenu = plusMenu.contains(event.target);
    const clickedPlusButton = plusButton.contains(event.target);

    if (!clickedInsideMenu && !clickedPlusButton) {
        plusMenu.classList.remove("open");
    }
});

sendButton.addEventListener("click", sendMessage);

messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

messageInput.addEventListener("input", () => {
    messageInput.style.height = "68px";
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 160)}px`;
});

function scrollToBottom() {
    chatPanel.scrollTop = chatPanel.scrollHeight;
}

function addMessage(content, type, options = {}) {
    const row = document.createElement("div");
    row.classList.add("message-row");

    if (type === "user") {
        row.classList.add("user-row");

        row.innerHTML = `
            <div class="message-bubble user-bubble">
                <div class="message-content"></div>
            </div>
        `;
    } else {
        row.classList.add("thux-row");

        row.innerHTML = `
            <div class="avatar">
                <img src="/frontend/assets/thux-logo.png" alt="Thux" />
            </div>

            <div class="message-bubble thux-bubble">
                <div class="message-content"></div>
            </div>
        `;
    }

    const contentElement = row.querySelector(".message-content");

    if (options.html) {
        contentElement.innerHTML = content;
    } else {
        contentElement.textContent = content;
    }

    chatPanel.appendChild(row);
    scrollToBottom();

    return row;
}

function createLoadingMessage() {
    return addMessage(
        `Pensando<span class="loading-dots"><span></span><span></span><span></span></span>`,
        "thux",
        {
            html: true,
        }
    );
}

function setSendingState(state) {
    isSending = state;
    sendButton.disabled = state;
    messageInput.disabled = state;
    sendButton.textContent = state ? "…" : "➤";
}

function getFriendlyErrorMessage() {
    return "Deu ruim ao falar com o Thux. Tenta de novo em alguns segundos.";
}

async function sendMessage() {
    const message = messageInput.value.trim();

    if (!message || isSending) {
        return;
    }

    addMessage(message, "user");

    messageInput.value = "";
    messageInput.style.height = "68px";
    plusMenu.classList.remove("open");

    const loadingMessage = createLoadingMessage();
    const loadingContent = loadingMessage.querySelector(".message-content");

    setSendingState(true);

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                message: message,
            }),
        });

        if (!response.ok) {
            throw new Error(`Erro HTTP ${response.status}`);
        }

        const data = await response.json();

        loadingContent.textContent = data.response || "Recebi uma resposta vazia do servidor.";
    } catch (error) {
        console.error(error);
        loadingContent.textContent = getFriendlyErrorMessage();
    } finally {
        setSendingState(false);
        messageInput.focus();
        scrollToBottom();
    }
}
