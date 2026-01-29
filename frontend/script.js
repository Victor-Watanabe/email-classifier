const submitBtn = document.getElementById("submitBtn");
const emailTextInput = document.getElementById("emailText");
const pdfFileInput = document.getElementById("pdfFile");
const resultArea = document.getElementById("result");

// URLs da API
const API_TEXT_URL = "http://localhost:8000/classify/text";
const API_PDF_URL = "http://localhost:8000/classify/pdf";

submitBtn.addEventListener("click", async () => {
  resultArea.textContent = "⏳ Processando...";

  const emailText = emailTextInput.value.trim();
  const pdfFile = pdfFileInput.files[0];

  // ======================
  // VALIDAÇÕES (FRONTEND)
  // ======================
  if (!emailText && !pdfFile) {
    resultArea.textContent = "❌ Informe um texto OU envie um PDF.";
    return;
  }

  if (emailText && pdfFile) {
    resultArea.textContent =
      "❌ Envie apenas TEXTO ou PDF, não os dois ao mesmo tempo.";
    return;
  }

  try {
    let response;

    // ======================
    // CASO 1: TEXTO
    // ======================
    if (emailText) {
      response = await fetch(API_TEXT_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: emailText,
        }),
      });
    }

    // ======================
    // CASO 2: PDF
    // ======================
    if (pdfFile) {
      const formData = new FormData();
      formData.append("file", pdfFile);

      response = await fetch(API_PDF_URL, {
        method: "POST",
        body: formData,
      });
    }

    // ======================
    // TRATAMENTO DA RESPOSTA
    // ======================
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || "Erro ao consumir API");
    }

    const data = await response.json();

    // Exibe resultado
    resultArea.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    console.error(error);
    resultArea.textContent =
      "❌ Erro ao processar a solicitação.\n\n" + error.message;
  }
});
