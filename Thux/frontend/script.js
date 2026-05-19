/*
    Script principal da interface do Thux.

    Funções deste arquivo:
    - abrir e fechar o menu do botão +
    - enviar mensagens para a rota /chat
    - mostrar mensagens do usuário e do Thux na tela
    - manter o chat rolando para a última mensagem
*/

const chatPanel = document.getElementById("chatPanel");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const plusButton = document.getElementById("plusButton");
const plusMenu = document.getElementById("plusMenu");


// Abre ou fecha o menu do botão +
plusButton.addEventListener("click", () => {
    plusMenu.classList.toggle("open");
});


// Fecha o menu + quando clicar fora dele
document.addEventListener("click", (event) => {
    const clickedInsideMenu = plusMenu.contains(event.target);
    const clickedPlusButton = plusButton.contains(event.target);

    if (!clickedInsideMenu && !clickedPlusButton) {
        plusMenu.classList.remove("open");
    }
});


// Envia mensagem ao clicar no botão Enviar
sendButton.addEventListener("click", sendMessage);


// Envia mensagem ao apertar Enter
// Shift + Enter quebra linha normalmente
messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});


// Ajusta a altura do textarea conforme o texto cresce
messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = `${messageInput.scrollHeight}px`;
});


function addMessage(author, content, type) {
    /*
        Adiciona uma mensagem no painel do chat.

        author: nome de quem enviou
        content: texto da mensagem
        type: "user" ou "thux"
    */

    const messageElement = document.createElement("div");
    messageElement.classList.add("message");

    if (type === "user") {
        messageElement.classList.add("user-message");
    } else {
        messageElement.classList.add("thux-message");
    }

    messageElement.innerHTML = `
        <div class="message-author">${author}</div>
        <div class="message-content"></div>
    `;

    const contentElement = messageElement.querySelector(".message-content");
    contentElement.textContent = content;

    chatPanel.appendChild(messageElement);

    // Mantém o chat sempre na última mensagem
    chatPanel.scrollTop = chatPanel.scrollHeight;
}


async function sendMessage() {
    /*
        Envia a mensagem do usuário para o backend do Thux.
    */

    const message = messageInput.value.trim();

    if (!message) {
        return;
    }

    // Mostra a mensagem do usuário na tela
    addMessage("Você", message, "user");

    // Limpa o campo de texto
    messageInput.value = "";
    messageInput.style.height = "auto";

    // Fecha o menu +, caso esteja aberto
    plusMenu.classList.remove("open");

    // Mostra uma mensagem temporária enquanto o Thux responde
    addMessage("Thux", "Pensando...", "thux");

    // Pega a última mensagem, que é o "Pensando..."
    const loadingMessage = chatPanel.lastElementChild;
    const loadingContent = loadingMessage.querySelector(".message-content");

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
            throw new Error("Erro na resposta do servidor.");
        }

        const data = await response.json();

        // Substitui "Pensando..." pela resposta real
        loadingContent.textContent = data.response;

    } catch (error) {
        loadingContent.textContent =
            "Deu ruim ao falar com o Thux. Provavelmente é API, servidor ou rota quebrando. Vamos caçar esse erro com calma.";
    }
}
