function exportCurrentConversationAsText() {
    const conversation = getCurrentConversation();

    if (!conversation || conversation.messages.length === 0) {
        showToast("Não há conversa para exportar ainda.");
        return;
    }

    const safeTitle = conversation.title
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .replace(/[^a-zA-Z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .toLowerCase();

    const createdAt = new Date(conversation.createdAt).toLocaleString("pt-BR");
    const updatedAt = new Date(conversation.updatedAt).toLocaleString("pt-BR");

    const messagesHtml = conversation.messages.map((message) => {
        const label = message.role === "user" ? "Você" : "Thux";
        const className = message.role === "user" ? "user-message" : "thux-message";

        return `
            <section class="message ${className}">
                <h3>${escapeHtml(label)}</h3>
                <div>${formatMessageForPdf(message.content)}</div>
            </section>
        `;
    }).join("");

    const printWindow = window.open("", "_blank");

    if (!printWindow) {
        showToast("O navegador bloqueou a janela de exportação.");
        return;
    }

    printWindow.document.write(`
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8" />
            <title>${escapeHtml(safeTitle || "conversa-thux")}</title>

            <style>
                * {
                    box-sizing: border-box;
                }

                body {
                    margin: 0;
                    padding: 36px;
                    color: #171717;
                    background: #f4f1e7;
                    font-family: Arial, Helvetica, sans-serif;
                    line-height: 1.45;
                }

                .page {
                    max-width: 820px;
                    margin: 0 auto;
                    padding: 34px;
                    background: #ffffff;
                    border: 1px solid #ddd6c4;
                    border-radius: 18px;
                }

                .header {
                    padding-bottom: 18px;
                    margin-bottom: 24px;
                    border-bottom: 3px solid #e0bd46;
                }

                .header h1 {
                    margin: 0;
                    font-size: 32px;
                    letter-spacing: -1px;
                }

                .header p {
                    margin: 6px 0 0;
                    color: #555;
                    font-size: 14px;
                }

                .meta {
                    margin-top: 14px;
                    padding: 12px 14px;
                    background: #f7f3e4;
                    border-radius: 12px;
                    font-size: 13px;
                    color: #333;
                }

                .message {
                    margin: 0 0 18px;
                    padding: 16px 18px;
                    border-radius: 14px;
                    page-break-inside: avoid;
                    border: 1px solid #e4e4e4;
                }

                .message h3 {
                    margin: 0 0 8px;
                    font-size: 15px;
                }

                .message div {
                    white-space: pre-wrap;
                    font-size: 14px;
                }

                .user-message {
                    background: #f1f1f1;
                }

                .thux-message {
                    background: #fff8df;
                    border-color: #ead37a;
                }

                .footer {
                    margin-top: 28px;
                    padding-top: 14px;
                    border-top: 1px solid #ddd;
                    color: #777;
                    font-size: 12px;
                    text-align: center;
                }

                @media print {
                    body {
                        background: #ffffff;
                        padding: 0;
                    }

                    .page {
                        border: none;
                        border-radius: 0;
                        max-width: none;
                    }
                }
            </style>
        </head>

        <body>
            <main class="page">
                <header class="header">
                    <h1>Thux-AI</h1>
                    <p>${escapeHtml(conversation.discipline)} · ${escapeHtml(conversation.lessonMode)}</p>

                    <div class="meta">
                        <strong>Título:</strong> ${escapeHtml(conversation.title)}<br />
                        <strong>Criada em:</strong> ${escapeHtml(createdAt)}<br />
                        <strong>Atualizada em:</strong> ${escapeHtml(updatedAt)}
                    </div>
                </header>

                ${messagesHtml}

                <footer class="footer">
                    Exportado pelo Thux-AI — desenvolvido por Iago Mesquita
                </footer>
            </main>

            <script>
                window.onload = function () {
                    window.print();
                };
            <\/script>
        </body>
        </html>
    `);

    printWindow.document.close();

    showToast("Abrindo exportação. Escolha 'Salvar como PDF'.");
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
