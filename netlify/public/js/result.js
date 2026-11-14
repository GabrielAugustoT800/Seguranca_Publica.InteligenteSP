// Exibe o resultado guardado pelo formulário. Usa apenas textContent (sem injeção de HTML).

// Lê o resultado salvo nesta aba.
const salvo = JSON.parse(sessionStorage.getItem('resultado') || 'null');
// Sem resultado (acesso direto à página), volta ao formulário.
if (!salvo) window.location.replace('form.html');
// Separa entrada e saída.
const { entrada, saida } = salvo || { entrada: {}, saida: {} };
// Formata uma probabilidade em porcentagem com uma casa decimal (padrão brasileiro).
const pct = (p) => (p * 100).toLocaleString('pt-BR', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + '%';

// Só preenche a página quando houver resultado.
if (salvo) {
  // Seleciona o elemento da classe.
  const classe = document.getElementById('classe');
  // Escreve a classe estimada.
  classe.textContent = saida.classe;
  // Aplica a cor da classe pelo CSS.
  classe.classList.add(saida.classe === 'Roubo' ? 'roubo' : 'furto');
  // Ajusta a largura da barra de roubo.
  document.getElementById('barra-roubo').style.width = pct(saida.prob_roubo).replace(',', '.');
  // Escreve a probabilidade de roubo.
  document.getElementById('valor-roubo').textContent = pct(saida.prob_roubo);
  // Ajusta a largura da barra de furto.
  document.getElementById('barra-furto').style.width = pct(saida.prob_furto).replace(',', '.');
  // Escreve a probabilidade de furto.
  document.getElementById('valor-furto').textContent = pct(saida.prob_furto);
  // Explica a regra de decisão.
  document.getElementById('limiar').textContent = `Regra de decisão: Roubo quando a probabilidade de roubo é maior ou igual a ${pct(saida.limiar)}.`;
  // Escreve o aviso de interpretação vindo da API.
  document.getElementById('interpretacao').textContent = saida.aviso_interpretacao;
  // Seleciona a lista de avisos.
  const lista = document.getElementById('avisos');
  // Cria um item para cada aviso (campos ausentes ou desconhecidos).
  (saida.avisos || []).forEach((a) => { const li = document.createElement('li'); li.textContent = a; lista.append(li); });
  // Mostra a lista só se houver avisos.
  lista.hidden = !(saida.avisos || []).length;
  // Define os rótulos do resumo na ordem do formulário.
  const rotulos = [['tipo_local', 'Tipo de local'], ['subtipo_local', 'Subtipo de local'], ['bairro', 'Bairro'], ['periodo', 'Período do dia'], ['data', 'Data'], ['hora', 'Hora']];
  // Seleciona a lista do resumo.
  const resumo = document.getElementById('resumo');
  // Acrescenta cada par rótulo/valor.
  rotulos.forEach(([k, r]) => {
    // Cria o termo (rótulo).
    const dt = document.createElement('dt');
    // Escreve o rótulo.
    dt.textContent = r;
    // Cria a descrição (valor informado).
    const dd = document.createElement('dd');
    // Escreve o valor ou "não informado".
    dd.textContent = (entrada[k] || '').trim() || 'não informado';
    // Acrescenta o par ao resumo.
    resumo.append(dt, dd);
  });
  // Mostra as versões do modelo e do contrato.
  document.getElementById('versao').textContent = `Modelo ${saida.versao_modelo} · contrato da API ${saida.versao_contrato}`;
  // Leva o foco ao título, anunciando a nova página a leitores de tela.
  document.getElementById('titulo-resultado').focus();
}
