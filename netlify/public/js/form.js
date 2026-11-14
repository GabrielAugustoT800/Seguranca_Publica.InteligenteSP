// Lógica do formulário: carrega opções, valida no navegador, chama a API e guarda o resultado.
import { classificarOcorrencia } from './api.js';

// Seleciona o formulário.
const form = document.getElementById('formulario');
// Seleciona o botão de envio (usado no estado de carregamento).
const botao = document.getElementById('enviar');
// Seleciona a caixa de erro geral (role="alert" anuncia a mensagem a leitores de tela).
const erroGeral = document.getElementById('erro-geral');
// Seleciona os campos de lista que serão preenchidos.
const selTipo = document.getElementById('tipo_local');
// Seleciona o campo de subtipo, dependente do tipo.
const selSubtipo = document.getElementById('subtipo_local');
// Seleciona o campo de período.
const selPeriodo = document.getElementById('periodo');
// Guarda as opções carregadas do arquivo estático.
let opcoes = null;

// Preenche um <select> com uma opção vazia e a lista de valores (textContent evita injeção de HTML).
function preencher(select, valores, rotuloVazio) {
  // Remove as opções anteriores.
  select.replaceChildren();
  // Cria a opção vazia, que significa "não informado".
  select.append(new Option(rotuloVazio, ''));
  // Acrescenta cada valor exatamente como o modelo o conhece.
  valores.forEach((v) => select.append(new Option(v.trim(), v)));
}

// Carrega as opções do formulário geradas a partir do treino de 2025.
async function carregarOpcoes() {
  try {
    // Busca o JSON estático de opções.
    const r = await fetch('data/opcoes.json');
    // Interrompe se o arquivo não foi encontrado.
    if (!r.ok) throw new Error();
    // Interpreta o JSON.
    opcoes = await r.json();
    // Preenche os tipos de local.
    preencher(selTipo, opcoes.tipos_local, 'Não informado');
    // Preenche os períodos do dia.
    preencher(selPeriodo, opcoes.periodos, 'Não informado');
    // Preenche as sugestões de bairro.
    document.getElementById('lista-bairros').replaceChildren(...opcoes.bairros.map((b) => new Option(b.trim(), b)));
  } catch {
    // Informa a falha sem bloquear o envio (campos vazios ainda são aceitos pela API).
    mostrarErro('Não foi possível carregar as listas de opções. Você ainda pode classificar com os campos disponíveis.');
  }
}

// Atualiza os subtipos quando o tipo de local muda.
selTipo.addEventListener('change', () => {
  // Obtém os subtipos do tipo escolhido (ou lista vazia).
  const lista = (opcoes && opcoes.subtipos_por_tipo[selTipo.value]) || [];
  // Preenche o campo de subtipo.
  preencher(selSubtipo, lista, selTipo.value ? 'Não informado' : 'Escolha o tipo primeiro');
});

// Ao limpar o formulário, também esvazia os subtipos e as mensagens.
form.addEventListener('reset', () => {
  // Restaura o campo de subtipo.
  preencher(selSubtipo, [], 'Escolha o tipo primeiro');
  // Esconde mensagens de erro.
  limparErros();
});

// Mostra a mensagem de erro geral.
function mostrarErro(texto) {
  // Define o texto da mensagem.
  erroGeral.textContent = texto;
  // Torna a caixa visível.
  erroGeral.hidden = false;
}

// Marca um campo como inválido, mostra a mensagem e devolve o elemento.
function marcarCampo(nome, texto) {
  // Seleciona o campo pelo nome.
  const campo = form.elements[nome];
  // Sinaliza o estado inválido para tecnologias assistivas.
  campo.setAttribute('aria-invalid', 'true');
  // Seleciona a mensagem específica do campo, se existir.
  const msg = document.getElementById(`erro-${nome}`);
  // Mostra a mensagem específica.
  if (msg) { msg.textContent = texto; msg.hidden = false; }
  // Devolve o campo para receber o foco.
  return campo;
}

// Remove todas as marcações de erro.
function limparErros() {
  // Esconde a caixa geral.
  erroGeral.hidden = true;
  // Remove aria-invalid dos campos.
  form.querySelectorAll('[aria-invalid]').forEach((c) => c.removeAttribute('aria-invalid'));
  // Esconde as mensagens por campo.
  form.querySelectorAll('.erro-campo').forEach((m) => { m.hidden = true; });
}

// Rótulos legíveis dos campos para as mensagens de erro.
const ROTULOS = { data: 'Data', hora: 'Hora', tipo_local: 'Tipo de local', subtipo_local: 'Subtipo', bairro: 'Bairro', periodo: 'Período' };

// Trata o envio do formulário.
form.addEventListener('submit', async (e) => {
  // Impede o envio tradicional (recarregaria a página).
  e.preventDefault();
  // Limpa erros anteriores.
  limparErros();
  // Lê os campos do formulário como objeto (todos textos; vazio = ausente).
  const campos = Object.fromEntries(new FormData(form).entries());
  // Rejeita datas que o navegador marcou como inválidas (ex.: digitação incompleta).
  if (form.elements.data.validity.badInput) { marcarCampo('data', 'Data incompleta ou inválida.').focus(); return; }
  // Rejeita horas incompletas.
  if (form.elements.hora.validity.badInput) { marcarCampo('hora', 'Hora incompleta ou inválida.').focus(); return; }
  // Entra no estado de carregamento.
  botao.disabled = true;
  // Indica o processamento no texto do botão.
  botao.textContent = 'Classificando…';
  // Marca o formulário como ocupado para leitores de tela.
  form.setAttribute('aria-busy', 'true');
  // Chama a API.
  const r = await classificarOcorrencia(campos);
  // Sai do estado de carregamento.
  botao.disabled = false;
  // Restaura o texto do botão.
  botao.textContent = 'Classificar';
  // Remove a marcação de ocupado.
  form.removeAttribute('aria-busy');
  // Em caso de sucesso, guarda resultado e entrada na sessão e abre a tela de resultado.
  if (r.ok) {
    // Guarda o resultado apenas nesta aba (sessionStorage some ao fechar).
    sessionStorage.setItem('resultado', JSON.stringify({ entrada: campos, saida: r.dados }));
    // Navega para o resultado.
    window.location.href = 'result.html';
    // Encerra o tratamento.
    return;
  }
  // Marca os campos que o servidor rejeitou.
  const invalidos = (r.erro.campos || []).filter((c) => form.elements[c]);
  // Aplica a marcação em cada campo inválido.
  invalidos.forEach((c) => marcarCampo(c, `${ROTULOS[c] || c} inválido.`));
  // Mostra a mensagem do servidor, com os campos envolvidos.
  mostrarErro(r.erro.mensagem + (invalidos.length ? ` Verifique: ${invalidos.map((c) => ROTULOS[c] || c).join(', ')}.` : ''));
  // Leva o foco ao primeiro campo inválido (ou à mensagem).
  (invalidos.length ? form.elements[invalidos[0]] : erroGeral).focus();
});

// Permite focar a caixa de erro por script.
erroGeral.tabIndex = -1;
// Inicia o carregamento das opções.
carregarOpcoes();
