const sidebar = document.querySelector(".sidebar");
const teacherTools = document.getElementById("teacherTools");
const sidebarDisciplineLabel = document.getElementById("sidebarDisciplineLabel");
const sidebarFunctionLabel = document.getElementById("sidebarFunctionLabel");
const newChatButton = document.getElementById("newChatButton");

const startScreen = document.getElementById("startScreen");
const functionScreen = document.getElementById("functionScreen");
const functionGrid = document.getElementById("functionGrid");
const functionTitle = document.getElementById("functionTitle");
const functionDescription = document.getElementById("functionDescription");
const backToStartButton = document.getElementById("backToStartButton");

const chatScreen = document.getElementById("chatScreen");
const chatPanel = document.getElementById("chatPanel");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const plusButton = document.getElementById("plusButton");
const plusMenu = document.getElementById("plusMenu");
const modeSubtitle = document.getElementById("modeSubtitle");
const backButton = document.getElementById("backButton");
const toast = document.getElementById("toast");

let isSending = false;
let selectedDiscipline = null;
let selectedLessonMode = null;

const lessonModes = {
    "Física": [
        {
            title: "Demonstrar conceito",
            description: "Explicar ideias físicas com exemplos de aula.",
        },
        {
            title: "Criar/resolver questão",
            description: "Gerar ou resolver problemas passo a passo.",
        },
        {
            title: "Modo prova/demonstração",
            description: "Deduzir fórmulas e justificar cada passagem.",
        },
        {
            title: "Buscar vídeo da biblioteca",
            description: "Encontrar cenas e vídeos para explicar Física com jogos.",
        },
        {
            title: "Planejar aula",
            description: "Organizar explicação, perguntas, exemplos e atividade.",
        },
    ],

    "Matemática": [
        {
            title: "Funções e álgebra",
            description: "Trabalhar funções, equações, domínio, imagem e gráficos.",
        },
        {
            title: "Geometria",
            description: "Explorar figuras, áreas, volumes, ângulos e relações.",
        },
        {
            title: "Criar/resolver exercício",
            description: "Gerar ou resolver exercícios com passo a passo.",
        },
        {
            title: "Modo prova/demonstração",
            description: "Demonstrar fórmulas, propriedades e resultados.",
        },
        {
            title: "Gerar gráfico",
            description: "Criar ou interpretar gráficos e representações visuais.",
        },
        {
            title: "Planejar aula",
            description: "Montar sequência didática, exemplos e fechamento.",
        },
    ],
};

document.querySelectorAll(".discipline-choice").forEach((button) => {
    button.addEventListener("click", () => {
        selectedDiscipline = button.dataset.discipline;
        openFunctionScreen(selectedDiscipline);
    });
});

backToStartButton.addEventListener("click", () => {
    functionScreen.classList.add("hidden");
    startScreen.classList.remove("hidden");
    selectedDiscipline = null;
    selectedLessonMode = null;
});

backButton.addEventListener("click", () => {
    chatScreen.classList.add("hidden");
    functionScreen.classList.remove("hidden");

    sidebar.classList.remove("chat-active");
    teacherTools.classList.add("hidden");

    selectedLessonMode = null;
    chatPanel.innerHTML = "";
});

newChatButton.addEventListener("click", () => {
    chatPanel.innerHTML = "";
    messageInput.focus();
});

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

document.querySelectorAll(".soon-button").forEach((button) => {
    button.addEventListener("click", () => {
        showToast("Função planejada para a próxima versão.");
    });
});

sendButton.addEventListener("click", sendMessage);

messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

messageInput.addEventListener("input", () => {
    messageInput.style.height = "62px";
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 150)}px`;
});

function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");

    setTimeout(() => {
        toast.classList.remove("show");
    }, 2200);
}

function openFunctionScreen(discipline) {
    startScreen.classList.add("hidden");
    functionScreen.classList.remove("hidden");

    functionTitle.textContent = `Como o Thux vai ajudar em ${discipline}?`;

    functionDescription.textContent =
        discipline === "Física"
            ? "Escolha uma função para orientar demonstrações, questões, vídeos e planejamento."
            : "Escolha uma função para orientar conceitos, geometria, gráficos, exercícios e demonstrações.";

    renderFunctionButtons(discipline);
}

function renderFunctionButtons(discipline) {
    functionGrid.innerHTML = "";

    lessonModes[discipline].forEach((mode) => {
        const button = document.createElement("button");
        button.type = "button";
        button.classList.add("function-button");
        button.dataset.mode = mode.title;

        button.innerHTML = `
            <strong>${mode.title}</strong>
            <span>${mode.description}</span>
        `;

        button.addEventListener("click", () => {
            selectedLessonMode = mode.title;
            openChatScreen();
        });

        functionGrid.appendChild(button);
    });
}

function openChatScreen() {
    functionScreen.classList.add("hidden");
    chatScreen.classList.remove("hidden");

    sidebar.classList.add("chat-active");
    teacherTools.classList.remove("hidden");

    sidebarDisciplineLabel.textContent = selectedDiscipline;
    sidebarFunctionLabel.textContent = selectedLessonMode;

    modeSubtitle.textContent = `${selectedDiscipline} · ${selectedLessonMode}`;

    chatPanel.innerHTML = "";
    messageInput.focus();
}

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
                <img src="/frontend/assets/thux-avatar.png" alt="Thux" />
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

async function sendMessage() {
    const message = messageInput.value.trim();

    if (!message || isSending) {
        return;
    }

    addMessage(message, "user");

    messageInput.value = "";
    messageInput.style.height = "62px";
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
                discipline: selectedDiscipline,
                lesson_mode: selectedLessonMode,
            }),
        });

        if (!response.ok) {
            throw new Error(`Erro HTTP ${response.status}`);
        }

        const data = await response.json();

        loadingContent.textContent =
            data.response || "Recebi uma resposta vazia do servidor.";
    } catch (error) {
        console.error(error);
        loadingContent.textContent =
            "Deu ruim ao falar com o Thux. Tenta de novo em alguns segundos.";
    } finally {
        setSendingState(false);
        messageInput.focus();
        scrollToBottom();
    }
}
