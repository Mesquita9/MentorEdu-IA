/*
    Script principal da interface do Thux.

    Funções:
    - abrir e fechar menu do botão +
    - enviar mensagens para /chat
    - renderizar mensagens
    - controlar loading
    - manter chat na última mensagem
*/

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
    messageInput.style.height = "auto";
    messageInput.style.height = `${messageInput.scrollHeight}px`;
});


function scrollToBottom() {
    chatPanel.scrollTop = chatPanel.scrollHeight;
}


function addMessage(author, content, type, options = {}) {
    const messageElement = document.createElement("div");
    messageElement.classList.add("message");

    if (type === "user") {
        messageElement.classList.add("user-message");
    } else {
        messageElement.classList.add("thux-message");
    }

    const tag = options.tag || (type === "user" ? "entrada" : "resposta");

    messageElement.innerHTML = `
        <div class="message-meta">
            <span class="message-author">${author}</span>
            <span class="message-tag">${tag}</span>
        </div>
        <div class="message-content"></div>
    `;

    const contentElement = messageElement.querySelector(".message-content");

    if (options.html) {
        contentElement.innerHTML = content;
    } else {
        contentElement.textContent = content;
    }

    chatPanel.appendChild(messageElement);
    scrollToBottom();

    return messageElement;
}


function createLoadingMessage() {
    return addMessage(
        "Thux",
        `
            <span>Consultando contexto</span>
            <span class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </span>
        `,
        "thux",
        {
            tag: "processando",
            html: true,
        }
    );
}


function setSendingState(state) {
    isSending = state;
    sendButton.disabled = state;
    messageInput.disabled = state;
    sendButton.querySelector("span").textContent = state ? "..." : "Enviar";
}


function getFriendlyErrorMessage() {
    return (
        "Deu ruim ao falar com o Thux. Pode ser API, servidor ou deploy ainda acordando. " +
        "Tenta de novo em alguns segundos; se persistir, a gente caça o erro nos logs."
    );
}


async function sendMessage() {
    const message = messageInput.value.trim();

    if (!message || isSending) {
        return;
    }

    addMessage("Você", message, "user");

    messageInput.value = "";
    messageInput.style.height = "auto";
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

        loadingMessage.querySelector(".message-tag").textContent = "resposta";
        loadingContent.textContent = data.response || "Recebi uma resposta vazia do servidor.";

    } catch (error) {
        console.error(error);

        loadingMessage.querySelector(".message-tag").textContent = "erro";
        loadingContent.textContent = getFriendlyErrorMessage();

    } finally {
        setSendingState(false);
        messageInput.focus();
        scrollToBottom();
    }
}
