// Cliente do contrato da API (v1.0). Isola a comunicação HTTP da interface,
// para que outro cliente (ex.: React Native) siga o mesmo contrato.

// Define o endereço da função de classificação no mesmo domínio do site.
const API_CLASSIFICAR = '/api/classificar';
// Define o tempo máximo de espera pela resposta, em milissegundos.
const TEMPO_LIMITE_MS = 15000;

// Envia os campos da ocorrência e devolve {ok, dados} ou {ok:false, erro:{codigo, mensagem, campos}}.
export async function classificarOcorrencia(campos) {
  // Cria um controlador para cancelar a requisição se ela demorar demais.
  const controle = new AbortController();
  // Agenda o cancelamento após o tempo limite.
  const timer = setTimeout(() => controle.abort(), TEMPO_LIMITE_MS);
  try {
    // Faz o POST com o corpo JSON definido no contrato.
    const resposta = await fetch(API_CLASSIFICAR, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(campos), signal: controle.signal });
    // Tenta interpretar o corpo como JSON (pode falhar em erros de infraestrutura).
    const corpo = await resposta.json().catch(() => null);
    // Sucesso: devolve os dados da classificação.
    if (resposta.ok && corpo) return { ok: true, dados: corpo };
    // Erro previsto no contrato: devolve o objeto de erro do servidor.
    if (corpo && corpo.erro) return { ok: false, erro: corpo.erro };
    // Erro sem corpo reconhecível: devolve mensagem genérica.
    return { ok: false, erro: { codigo: 'falha_servidor', mensagem: 'O serviço não respondeu como esperado. Tente novamente.' } };
  } catch (e) {
    // Diferencia tempo esgotado de falha de rede.
    const codigo = e.name === 'AbortError' ? 'tempo_esgotado' : 'falha_rede';
    // Devolve mensagem amigável conforme o caso.
    return { ok: false, erro: { codigo, mensagem: codigo === 'tempo_esgotado' ? 'O serviço demorou demais para responder.' : 'Não foi possível conectar ao serviço. Verifique sua conexão.' } };
  } finally {
    // Cancela o temporizador em qualquer desfecho.
    clearTimeout(timer);
  }
}
