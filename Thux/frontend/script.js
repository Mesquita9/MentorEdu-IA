const sidebar = document.querySelector(".sidebar");
const teacherTools = document.getElementById("teacherTools");
const sidebarDisciplineLabel = document.getElementById("sidebarDisciplineLabel");
const sidebarFunctionLabel = document.getElementById("sidebarFunctionLabel");

const newChatButton = document.getElementById("newChatButton");
const saveChatButton = document.getElementById("saveChatButton");
const exportPdfButton = document.getElementById("exportPdfButton");
const deleteChatButton = document.getElementById("deleteChatButton");

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

const STORAGE_KEY = "thux_conversations_v1";

let isSending = false;
let selectedDiscipline = null;
let selectedArea = null;
let selectedLessonMode = null;
let currentConversationId = null;
let conversations = loadConversations();

const mathAreas = [
    {
        title: "Conjuntos e funções",
        description: "Conjuntos, relações, domínio, imagem, funções e gráficos iniciais.",
        libraryId: "matematica_conjuntos_funcoes_001",
    },
    {
        title: "Logaritmos",
        description: "Definição, propriedades, equações logarítmicas e aplicações.",
        libraryId: "matematica_logaritmos_001",
    },
    {
        title: "Trigonometria",
        description: "Razões trigonométricas, ciclo, identidades e funções trigonométricas.",
        libraryId: "matematica_trigonometria_001",
    },
    {
        title: "Sequências, matrizes e sistemas",
        description: "Progressões, matrizes, determinantes e sistemas lineares.",
        libraryId: "matematica_sequencias_matrizes_sistemas_001",
    },
    {
        title: "Combinatória e probabilidade",
        description: "Contagem, arranjos, combinações, permutações e probabilidade.",
        libraryId: "matematica_combinatoria_probabilidade_001",
    },
    {
        title: "Complexos e polinômios",
        description: "Números complexos, equações polinomiais e propriedades algébricas.",
        libraryId: "matematica_complexos_polinomios_001",
    },
    {
        title: "Geometria analítica",
        description: "Plano cartesiano, reta, circunferência, cônicas e distância.",
        libraryId: "matematica_geometria_analitica_001",
    },
    {
        title: "Limites, derivadas e integral",
        description: "Noções iniciais de cálculo: limites, derivadas e integrais.",
        libraryId: "matematica_limites_derivadas_integral_001",
    },
    {
        title: "Geometria plana",
        description: "Ângulos, polígonos, circunferência, áreas e relações métricas.",
        libraryId: "matematica_geometria_plana_001",
    },
    {
        title: "Geometria espacial",
        description: "Prismas, pirâmides, cilindros, cones, esferas, áreas e volumes.",
        libraryId: "matematica_geometria_espacial_001",
    },
    {
        title: "Financeira e estatística",
        description: "Porcentagem, juros, médias, gráficos estatísticos e análise de dados.",
        libraryId: "matematica_financeira_estatistica_001",
    },
];

const defaultStudyTools = [
    {
        title: "Explicar conceito",
        description: "Explicar a teoria com base no material cadastrado da área.",
    },
    {
        title: "Resolver questão",
        description: "Resolver uma questão passo a passo usando a base correta.",
    },
    {
        title: "Modo prova/demonstração",
        description: "Justificar propriedades, fórmulas e passagens com mais rigor.",
    },
    {
        title: "Buscar referência no livro",
        description: "Procurar trecho, página ou definição no material da biblioteca.",
    },
];

const graphTool = {
    title: "Gerar gráfico",
    description: "Criar e analisar gráficos com pontos, raízes, vértice e comportamento.",
};

const sectionTools = {
    "Planejamento": [
        {
            title: "Planejar aula",
            description: "Criar uma sequência didática com objetivos, explicação e atividade.",
        },
        {
            title: "Criar lista de exercícios",
            description: "Montar uma lista por tema, nível e objetivo da aula.",
        },
        {
            title: "Criar avaliação",
            description: "Gerar uma avaliação com questões, critérios e gabarito.",
        },
        {
            title: "Roteiro de demonstração",
            description: "Planejar uma demonstração prática ou visual para sala.",
        },
    ],

    "Física": [
        {
            title: "Mecânica básica",
            description: "Trabalhar conceitos iniciais de movimento, força, energia e leis de Newton.",
        },
        {
            title: "Resolver questão",
            description: "Resolver problemas de Física passo a passo, com unidades e interpretação.",
        },
        {
            title: "Demonstrar conceito",
            description: "Explicar conceitos físicos com analogias, exemplos e linguagem de aula.",
        },
        {
            title: "Buscar vídeo da biblioteca",
            description: "Encontrar vídeos, cenas ou gravações organizadas por tema.",
        },
    ],

    "Linguagem": [
        {
            title: "Interpretação de questões",
            description: "Ajudar a ler enunciados, comandos e pegadinhas de prova.",
        },
        {
            title: "Símbolos matemáticos",
            description: "Explicar símbolos como ∈, ∀, ∃, ⇒, ⇔, reticências e notações.",
        },
        {
            title: "Conectivos lógicos",
            description: "Trabalhar 'se', 'somente se', 'se e somente se', negação e equivalências.",
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
    saveCurrentConversation();

    if (selectedDiscipline === "Matemática" && selectedArea) {
        selectedArea = null;
        selectedLessonMode = null;
        renderMathAreas();
        return;
    }

    functionScreen.classList.add("hidden");
    startScreen.classList.remove("hidden");

    selectedDiscipline = null;
    selectedArea = null;
    selectedLessonMode = null;
    currentConversationId = null;

    resetChatUiState();
});

backButton.addEventListener("click", () => {
    saveCurrentConversation();

    chatScreen.classList.add("hidden");
    functionScreen.classList.remove("hidden");

    sidebar.classList.remove("chat-active");
    teacherTools.classList.add("hidden");

    selectedLessonMode = null;
    currentConversationId = null;
    chatPanel.innerHTML = "";

    resetChatUiState();

    if (selectedDiscipline === "Matemática" && selectedArea) {
        renderMathTools(selectedArea);
    } else {
        renderSectionTools(selectedDiscipline);
    }
});

newChatButton.addEventListener("click", () => {
    createNewConversation(true);
});

saveChatButton.addEventListener("click", () => {
    saveCurrentConversation();
    showToast("Conversa salva neste navegador.");
});

deleteChatButton.addEventListener("click", () => {
    deleteCurrentConversation();
});

exportPdfButton.addEventListener("click", () => {
    exportCurrentConversationAsPdf();
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

document.querySelectorAll(".attachment-soon-button").forEach((button) => {
    button.addEventListener("click", () => {
        plusMenu.classList.remove("open");
        showToast("Anexos entram na próxima etapa.");
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

function resetChatUiState() {
    isSending = false;
    sendButton.disabled = false;
    messageInput.disabled = false;
    sendButton.textContent = "➤";
    messageInput.value = "";
    messageInput.style.height = "62px";
    plusMenu.classList.remove("open");
}

function loadConversations() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        return saved ? JSON.parse(saved) : [];
    } catch (error) {
        console.error("Erro ao carregar conversas:", error);
        return [];
    }
}

function persistConversations() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

function generateId() {
    if (window.crypto && crypto.randomUUID) {
        return crypto.randomUUID();
    }

    return `conv-${Date.now()}-${Math.floor(Math.random() * 100000)}`;
}

function getModeKey(discipline, area, lessonMode) {
    return `${discipline}::${area || "geral"}::${lessonMode}`;
}

function getCurrentConversation() {
    return conversations.find((conversation) => conversation.id === currentConversationId) || null;
}

function findLastConversationForMode(discipline, area, lessonMode) {
    const modeKey = getModeKey(discipline, area, lessonMode);

    return conversations
        .filter((conversation) => conversation.modeKey === modeKey)
        .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))[0] || null;
}

function createConversationObject(discipline, area, lessonMode) {
    const now = new Date().toISOString();
    const title = area
        ? `${discipline} - ${area} - ${lessonMode}`
        : `${discipline} - ${lessonMode}`;

    return {
        id: generateId(),
        title,
        discipline,
        area,
        lessonMode,
        modeKey: getModeKey(discipline, area, lessonMode),
        messages: [],
        createdAt: now,
        updatedAt: now,
    };
}

function createNewConversation(showMessage = false) {
    if (!selectedDiscipline || !selectedLessonMode) {
        showToast("Escolha uma área e uma função primeiro.");
        return;
    }

    saveCurrentConversation();
    resetChatUiState();

    const conversation = createConversationObject(
        selectedDiscipline,
        selectedArea,
        selectedLessonMode
    );

    conversations.push(conversation);
    currentConversationId = conversation.id;

    persistConversations();
    renderConversation(conversation);

    setTimeout(() => {
        messageInput.focus();
    }, 0);

    if (showMessage) {
        showToast("Nova conversa criada.");
    }
}

function saveCurrentConversation() {
    const conversation = getCurrentConversation();

    if (!conversation) {
        return;
    }

    conversation.updatedAt = new Date().toISOString();
    persistConversations();
}

function deleteCurrentConversation() {
    const conversation = getCurrentConversation();

    if (!conversation) {
        showToast("Nenhuma conversa ativa para excluir.");
        return;
    }

    const shouldDelete = confirm("Excluir esta conversa? Essa ação remove o chat salvo neste navegador.");

    if (!shouldDelete) {
        return;
    }

    conversations = conversations.filter((item) => item.id !== conversation.id);
    persistConversations();

    currentConversationId = null;
    chatPanel.innerHTML = "";
    resetChatUiState();

    createNewConversation(false);
    showToast("Conversa excluída.");
}

function appendMessageToConversation(role, content) {
    const conversation = getCurrentConversation();

    if (!conversation) {
        return;
    }

    conversation.messages.push({
        role,
        content,
        createdAt: new Date().toISOString(),
    });

    conversation.updatedAt = new Date().toISOString();
    persistConversations();
}

function renderConversation(conversation) {
    chatPanel.innerHTML = "";

    conversation.messages.forEach((message) => {
        const type = message.role === "user" ? "user" : "thux";
        addMessage(message.content, type, { save: false });
    });

    scrollToBottom();
}

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

    selectedArea = null;
    selectedLessonMode = null;

    if (discipline === "Matemática") {
        renderMathAreas();
        return;
    }

    renderSectionTools(discipline);
}

function renderMathAreas() {
    backToStartButton.textContent = "← voltar";

    functionTitle.textContent = "Qual área da Matemática vamos trabalhar?";

    functionDescription.textContent =
        "Escolha a área conforme os livros cadastrados na biblioteca. O Thux deve buscar primeiro no material correspondente.";

    functionGrid.innerHTML = "";

    mathAreas.forEach((area) => {
        const button = document.createElement("button");
        button.type = "button";
        button.classList.add("function-button", "area-button");

        button.innerHTML = `
            <strong>${area.title}</strong>
            <span>${area.description}</span>
        `;

        button.addEventListener("click", () => {
            selectedArea = area.title;
            renderMathTools(area.title);
        });

        functionGrid.appendChild(button);
    });
}

function renderMathTools(areaTitle) {
    backToStartButton.textContent = "← áreas";

    functionTitle.textContent = areaTitle;

    functionDescription.textContent =
        "Agora escolha como o Thux deve trabalhar essa área usando a biblioteca cadastrada.";

    functionGrid.innerHTML = "";

    const tools = getToolsForMathArea(areaTitle);

    tools.forEach((tool) => {
        const button = document.createElement("button");
        button.type = "button";
        button.classList.add("function-button", "tool-button-card");

        button.innerHTML = `
            <strong>${tool.title}</strong>
            <span>${tool.description}</span>
        `;

        button.addEventListener("click", () => {
            selectedLessonMode = tool.title;
            openChatScreen();
        });

        functionGrid.appendChild(button);
    });
}

function getToolsForMathArea(areaTitle) {
    const tools = [...defaultStudyTools];

    const areasWithGraph = [
        "Conjuntos e funções",
        "Trigonometria",
        "Geometria analítica",
        "Limites, derivadas e integral",
    ];

    if (areasWithGraph.includes(areaTitle)) {
        tools.splice(2, 0, graphTool);
    }

    return tools;
}

function renderSectionTools(discipline) {
    backToStartButton.textContent = "← voltar";

    functionTitle.textContent = `Como o Thux vai trabalhar ${discipline}?`;

    if (discipline === "Planejamento") {
        functionDescription.textContent =
            "Ambiente para preparar aulas, listas, avaliações e roteiros fora do momento de sala.";
    } else if (discipline === "Física") {
        functionDescription.textContent =
            "Escolha uma frente de Física. Por enquanto, vamos manter simples até a biblioteca de Física estar organizada.";
    } else if (discipline === "Linguagem") {
        functionDescription.textContent =
            "Use Linguagem para interpretação de questões, símbolos matemáticos e conectivos lógicos.";
    } else {
        functionDescription.textContent =
            "Escolha uma função para orientar a resposta do Thux.";
    }

    functionGrid.innerHTML = "";

    const tools = sectionTools[discipline] || [];

    tools.forEach((tool) => {
        const button = document.createElement("button");
        button.type = "button";
        button.classList.add("function-button", "tool-button-card");

        button.innerHTML = `
            <strong>${tool.title}</strong>
            <span>${tool.description}</span>
        `;

        button.addEventListener("click", () => {
            selectedArea = null;
            selectedLessonMode = tool.title;
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

    const activeContext = selectedArea
        ? `${selectedDiscipline} · ${selectedArea}`
        : selectedDiscipline;

    sidebarDisciplineLabel.textContent = activeContext;
    sidebarFunctionLabel.textContent = selectedLessonMode;
    modeSubtitle.textContent = `${activeContext} · ${selectedLessonMode}`;

    resetChatUiState();

    const lastConversation = findLastConversationForMode(
        selectedDiscipline,
        selectedArea,
        selectedLessonMode
    );

    if (lastConversation) {
        currentConversationId = lastConversation.id;
        renderConversation(lastConversation);
    } else {
        createNewConversation(false);
    }

    setTimeout(() => {
        messageInput.focus();
    }, 0);
}

function scrollToBottom() {
    chatPanel.scrollTop = chatPanel.scrollHeight;
}

function addMessage(content, type, options = {}) {
    const shouldSave = options.save !== false;

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

    if (shouldSave && !options.html) {
        appendMessageToConversation(type === "user" ? "user" : "assistant", content);
    }

    return row;
}

function updateAssistantMessage(row, content) {
    const contentElement = row.querySelector(".message-content");
    contentElement.textContent = content;

    appendMessageToConversation("assistant", content);
    scrollToBottom();
}

function createLoadingMessage() {
    return addMessage(
        `Pensando<span class="loading-dots"><span></span><span></span><span></span></span>`,
        "thux",
        {
            html: true,
            save: false,
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

    if (!currentConversationId) {
        createNewConversation(false);
    }

    addMessage(message, "user");

    messageInput.value = "";
    messageInput.style.height = "62px";
    plusMenu.classList.remove("open");

    const loadingMessage = createLoadingMessage();

    setSendingState(true);

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                message,
                discipline: selectedDiscipline,
                area: selectedArea,
                lesson_mode: selectedLessonMode,
                conversation_id: currentConversationId,
            }),
        });

        if (!response.ok) {
            throw new Error(`Erro HTTP ${response.status}`);
        }

        const data = await response.json();

        updateAssistantMessage(
            loadingMessage,
            data.response || "Recebi uma resposta vazia do servidor."
        );
    } catch (error) {
        console.error(error);

        updateAssistantMessage(
            loadingMessage,
            "Deu ruim ao falar com o Thux. Tenta de novo em alguns segundos."
        );
    } finally {
        setSendingState(false);
        messageInput.focus();
        scrollToBottom();
    }
}

function exportCurrentConversationAsPdf() {
    const conversation = getCurrentConversation();

    if (!conversation || conversation.messages.length === 0) {
        showToast("Não há conversa para exportar ainda.");
        return;
    }

    saveCurrentConversation();
    plusMenu.classList.remove("open");

    const createdAt = new Date(conversation.createdAt).toLocaleString("pt-BR");
    const updatedAt = new Date(conversation.updatedAt).toLocaleString("pt-BR");
    const summary = buildConversationSummary(conversation);
    const references = buildReferencesBlock(conversation);
    const messagesHtml = buildMessagesHtml(conversation);

    const oldPrintRoot = document.getElementById("thuxPrintRoot");
    const oldPrintStyle = document.getElementById("thuxPrintStyle");
    const oldPrintActions = document.getElementById("thuxPrintActions");

    if (oldPrintRoot) oldPrintRoot.remove();
    if (oldPrintStyle) oldPrintStyle.remove();
    if (oldPrintActions) oldPrintActions.remove();

    const printStyle = document.createElement("style");
    printStyle.id = "thuxPrintStyle";
    printStyle.textContent = `
        #thuxPrintRoot {
            position: fixed;
            inset: 0;
            z-index: 9998;
            overflow-y: auto;
            padding: 34px;
            background: #ffffff;
            color: #171717;
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.45;
        }

        #thuxPrintRoot * {
            box-sizing: border-box;
        }

        #thuxPrintRoot .print-page {
            max-width: 850px;
            margin: 0 auto 110px;
            background: #ffffff;
        }

        #thuxPrintRoot .cover {
            min-height: 78vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            border-bottom: 5px solid #e0bd46;
            margin-bottom: 34px;
        }

        #thuxPrintRoot .cover-badge {
            width: fit-content;
            margin-bottom: 18px;
            padding: 8px 12px;
            border-radius: 999px;
            background: #e0bd46;
            color: #171717;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        #thuxPrintRoot .cover h1 {
            margin: 0;
            font-size: 54px;
            letter-spacing: -2px;
        }

        #thuxPrintRoot .cover h2 {
            margin: 8px 0 0;
            color: #555;
            font-size: 20px;
            font-weight: 500;
        }

        #thuxPrintRoot .cover-meta {
            margin-top: 34px;
            padding: 16px;
            background: #f7f3e4;
            border-radius: 14px;
            font-size: 14px;
            color: #333;
        }

        #thuxPrintRoot .section-title {
            margin: 0 0 18px;
            padding-bottom: 8px;
            border-bottom: 3px solid #e0bd46;
            font-size: 26px;
            letter-spacing: -0.04em;
        }

        #thuxPrintRoot .content-section {
            margin-bottom: 28px;
        }

        #thuxPrintRoot .message {
            margin: 0 0 18px;
            padding: 16px 18px;
            border-radius: 14px;
            page-break-inside: avoid;
            border: 1px solid #e4e4e4;
        }

        #thuxPrintRoot .message h3 {
            margin: 0 0 8px;
            font-size: 15px;
        }

        #thuxPrintRoot .message div {
            white-space: pre-wrap;
            font-size: 14px;
        }

        #thuxPrintRoot .user-message {
            background: #f1f1f1;
        }

        #thuxPrintRoot .thux-message {
            background: #fff8df;
            border-color: #ead37a;
        }

        #thuxPrintRoot .summary-box,
        #thuxPrintRoot .references-box {
            padding: 16px 18px;
            border-radius: 14px;
            background: #f7f3e4;
            border: 1px solid #ead37a;
            font-size: 14px;
            white-space: pre-wrap;
        }

        #thuxPrintRoot .footer {
            margin-top: 34px;
            padding-top: 14px;
            border-top: 1px solid #ddd;
            color: #777;
            font-size: 12px;
            text-align: center;
        }

        #thuxPrintActions {
            position: fixed;
            left: 50%;
            bottom: 24px;
            transform: translateX(-50%);
            z-index: 9999;
            display: flex;
            gap: 12px;
            padding: 12px;
            border-radius: 18px;
            background: rgba(20, 20, 20, 0.92);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.35);
        }

        #thuxPrintActions button {
            height: 44px;
            padding: 0 18px;
            border: none;
            border-radius: 12px;
            font-weight: 850;
            cursor: pointer;
        }

        #thuxPrintSaveButton {
            background: #e0bd46;
            color: #171717;
        }

        #thuxPrintCancelButton {
            background: #eeeeee;
            color: #222222;
        }

        @media print {
            .app-layout,
            #thuxPrintActions {
                display: none !important;
            }

            body {
                overflow: visible !important;
                background: #ffffff !important;
            }

            #thuxPrintRoot {
                position: static !important;
                inset: auto !important;
                z-index: auto !important;
                overflow: visible !important;
                padding: 0 !important;
                background: #ffffff !important;
            }

            #thuxPrintRoot .print-page {
                max-width: none !important;
                margin: 0 !important;
            }

            #thuxPrintRoot .cover {
                min-height: 86vh;
                page-break-after: always;
            }
        }
    `;

    const printRoot = document.createElement("section");
    printRoot.id = "thuxPrintRoot";

    printRoot.innerHTML = `
        <main class="print-page">
            <section class="cover">
                <div class="cover-badge">Relatório de conversa</div>

                <h1>Thux-AI</h1>
                <h2>${escapeHtml(conversation.discipline)}${conversation.area ? " · " + escapeHtml(conversation.area) : ""} · ${escapeHtml(conversation.lessonMode)}</h2>

                <div class="cover-meta">
                    <strong>Título:</strong> ${escapeHtml(conversation.title)}<br />
                    <strong>Criada em:</strong> ${escapeHtml(createdAt)}<br />
                    <strong>Atualizada em:</strong> ${escapeHtml(updatedAt)}<br />
                    <strong>Desenvolvido por:</strong> Iago Mesquita
                </div>
            </section>

            <section class="content-section">
                <h2 class="section-title">Conversa</h2>
                ${messagesHtml}
            </section>

            <section class="content-section">
                <h2 class="section-title">Resumo</h2>
                <div class="summary-box">${formatMessageForPdf(summary)}</div>
            </section>

            <section class="content-section">
                <h2 class="section-title">Referências</h2>
                <div class="references-box">${formatMessageForPdf(references)}</div>
            </section>

            <footer class="footer">
                Exportado pelo Thux-AI — desenvolvido por Iago Mesquita
            </footer>
        </main>
    `;

    const printActions = document.createElement("div");
    printActions.id = "thuxPrintActions";
    printActions.innerHTML = `
        <button type="button" id="thuxPrintSaveButton">Salvar como PDF</button>
        <button type="button" id="thuxPrintCancelButton">Cancelar</button>
    `;

    document.head.appendChild(printStyle);
    document.body.appendChild(printRoot);
    document.body.appendChild(printActions);

    document.getElementById("thuxPrintSaveButton").addEventListener("click", () => {
        window.print();
    });

    document.getElementById("thuxPrintCancelButton").addEventListener("click", () => {
        printRoot.remove();
        printActions.remove();
        printStyle.remove();
        messageInput.focus();
    });

    showToast("Prévia do PDF aberta.");
}

function buildMessagesHtml(conversation) {
    return conversation.messages.map((message) => {
        const label = message.role === "user" ? "Você" : "Thux";
        const className = message.role === "user" ? "user-message" : "thux-message";

        return `
            <section class="message ${className}">
                <h3>${escapeHtml(label)}</h3>
                <div>${formatMessageForPdf(message.content)}</div>
            </section>
        `;
    }).join("");
}

function buildConversationSummary(conversation) {
    const userMessages = conversation.messages.filter((message) => message.role === "user");
    const assistantMessages = conversation.messages.filter((message) => message.role === "assistant");

    const firstUserMessage = userMessages[0]?.content || "Sem pergunta inicial registrada.";
    const lastAssistantMessage = assistantMessages[assistantMessages.length - 1]?.content || "";

    const shortFirstMessage = cutText(firstUserMessage, 260);
    const shortLastAnswer = cutText(lastAssistantMessage, 420);

    return [
        `Disciplina: ${conversation.discipline}`,
        conversation.area ? `Área: ${conversation.area}` : null,
        `Função: ${conversation.lessonMode}`,
        "",
        "Síntese provisória:",
        `A conversa começou a partir da seguinte demanda do usuário: "${shortFirstMessage}"`,
        "",
        shortLastAnswer
            ? `Último encaminhamento do Thux: ${shortLastAnswer}`
            : "Ainda não há resposta suficiente do Thux para gerar uma síntese mais detalhada.",
        "",
        "Observação: este resumo ainda é local e provisório. Futuramente será gerado pelo próprio Thux com base no conteúdo completo da conversa.",
    ].filter(Boolean).join("\n");
}

function buildReferencesBlock(conversation) {
    const possibleSourceLines = conversation.messages
        .filter((message) => message.role === "assistant")
        .flatMap((message) => {
            return message.content
                .split("\n")
                .filter((line) => {
                    const normalized = line.toLowerCase();

                    return (
                        normalized.includes("fonte") ||
                        normalized.includes("referência") ||
                        normalized.includes("referencia") ||
                        normalized.includes("página") ||
                        normalized.includes(".pdf")
                    );
                });
        });

    if (possibleSourceLines.length === 0) {
        return [
            "Nenhuma referência específica foi registrada automaticamente nesta conversa.",
            "",
            "Quando a resposta usar a biblioteca do Thux, esta seção deverá listar:",
            "- nome do arquivo/PDF;",
            "- página consultada;",
            "- disciplina;",
            "- área;",
            "- uso da fonte na resposta.",
        ].join("\n");
    }

    const uniqueLines = [...new Set(possibleSourceLines)];

    return uniqueLines.map((line) => `- ${line.trim()}`).join("\n");
}

function cutText(text, maxLength) {
    const cleanText = String(text).replace(/\s+/g, " ").trim();

    if (cleanText.length <= maxLength) {
        return cleanText;
    }

    return `${cleanText.slice(0, maxLength).trim()}...`;
}

function escapeHtml(text) {
    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatMessageForPdf(text) {
    return escapeHtml(text).replace(/\n/g, "<br />");
}
