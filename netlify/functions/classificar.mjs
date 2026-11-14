/**
 * Função serverless (Netlify Functions v2) que classifica uma ocorrência como Roubo ou Furto.
 * Reproduz em JavaScript puro o pré-processamento do treino (one-hot + mediana + StandardScaler)
 * e o forward pass da MLP 64→32→1 exportada por src/webapp/export_artifacts.py.
 *
 * Contrato v1.0 — POST /api/classificar (application/json, máx. 8 kB)
 *   Entrada: { tipo_local, subtipo_local, bairro, periodo, data: "AAAA-MM-DD", hora: "HH:MM" }
 *            (todos opcionais; vazio = ausente, imputado como no treino)
 *   200: { classe, prob_roubo, prob_furto, limiar, versao_modelo, versao_contrato, avisos, aviso_interpretacao }
 *   400: { erro: { codigo, mensagem, campos? } }   405: método não permitido   413: corpo grande demais
 *   500: { erro: { codigo: "erro_interno", mensagem } }  (sem detalhes internos)
 */

// Importa os pesos e vieses das camadas lineares da MLP treinada.
import pesos from './artefatos/pesos.mjs';
// Importa categorias do OneHotEncoder, medianas e parâmetros do StandardScaler.
import preprocesso from './artefatos/preprocesso.mjs';
// Importa os metadados públicos do modelo (limiar, versão, métricas).
import metadados from './artefatos/metadados.mjs';

// Define a rota pública da função (dispensa o redirecionamento do netlify.toml).
export const config = { path: '/api/classificar' };

// Limiar de decisão: prob_roubo >= LIMIAR resulta na classe Roubo.
export const LIMIAR = Number(metadados.threshold);
// Tamanho máximo aceito para o corpo da requisição, em bytes.
const LIMITE_BYTES = 8 * 1024;
// Tamanho máximo aceito para cada campo de texto.
const LIMITE_CAMPO = 200;
// Associa cada campo do contrato à coluna categórica usada no treino.
const CAMPOS_CATEGORICOS = { tipo_local: 'DESCR_TIPOLOCAL', subtipo_local: 'DESCR_SUBTIPOLOCAL', bairro: 'BAIRRO', periodo: 'DESCR_PERIODO' };
// Lista todos os campos aceitos pelo contrato (os demais são rejeitados).
const CAMPOS_ACEITOS = [...Object.keys(CAMPOS_CATEGORICOS), 'data', 'hora'];
// Pré-calcula, por coluna, um mapa categoria → posição no one-hot (busca O(1)).
const INDICES = Object.fromEntries(preprocesso.categoricas.map((col) => [col, new Map(preprocesso.categorias[col].map((c, i) => [c, i]))]));
// Texto explicativo exigido pelo enunciado sobre o significado da saída.
const AVISO = 'A saída é a classe estimada (roubo ou furto) para a ocorrência descrita. Não é uma estimativa de risco ou de chance de crime no local, na data ou no horário.';

// Cria uma resposta JSON com o status indicado e sem cache.
function json(status, corpo) {
  // Devolve o objeto Response serializado em UTF-8.
  return new Response(JSON.stringify(corpo), { status, headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' } });
}

// Cria uma resposta de erro no formato padronizado do contrato.
function erro(status, codigo, mensagem, campos) {
  // Monta o corpo { erro: {...} }, incluindo os campos inválidos quando houver.
  return json(status, { erro: { codigo, mensagem, ...(campos ? { campos } : {}) } });
}

// Valida a data AAAA-MM-DD e devolve {dia_semana (0=segunda), mes}; null se ausente; lança se inválida.
export function lerData(texto) {
  // Trata texto vazio como ausência (será imputado pela mediana).
  if (texto === '') return null;
  // Confere o formato AAAA-MM-DD.
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(texto);
  // Rejeita formatos diferentes.
  if (!m) throw new Error('data');
  // Monta a data em UTC para não depender do fuso do servidor.
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  // Rejeita datas inexistentes (ex.: 2025-02-30, que o Date "corrigiria").
  if (d.getUTCMonth() !== +m[2] - 1 || d.getUTCDate() !== +m[3]) throw new Error('data');
  // Converte o dia da semana do JS (0=domingo) para o do pandas (0=segunda).
  return { dia_semana: (d.getUTCDay() + 6) % 7, mes: d.getUTCMonth() + 1 };
}

// Valida a hora HH:MM (ou HH:MM:SS) e devolve a hora inteira 0–23; null se ausente; lança se inválida.
export function lerHora(texto) {
  // Trata texto vazio como ausência.
  if (texto === '') return null;
  // Confere o formato de hora.
  const m = /^(\d{2}):(\d{2})(:\d{2})?$/.exec(texto);
  // Rejeita formato inválido, hora acima de 23 ou minuto acima de 59.
  if (!m || +m[1] > 23 || +m[2] > 59) throw new Error('hora');
  // Devolve só a hora, como parse_hora do treino.
  return +m[1];
}

// Valida o corpo recebido e devolve os campos normalizados ou a lista de campos inválidos.
export function validar(corpo) {
  // Exige um objeto JSON simples.
  if (corpo === null || typeof corpo !== 'object' || Array.isArray(corpo)) return { invalidos: ['corpo'] };
  // Acumula os nomes de campos com problema.
  const invalidos = Object.keys(corpo).filter((k) => !CAMPOS_ACEITOS.includes(k));
  // Guarda os valores normalizados.
  const dados = {};
  // Percorre cada campo aceito pelo contrato.
  for (const campo of CAMPOS_ACEITOS) {
    // Lê o valor; ausente ou null vira texto vazio.
    const valor = corpo[campo] ?? '';
    // Exige texto de tamanho limitado.
    if (typeof valor !== 'string' || valor.length > LIMITE_CAMPO) { invalidos.push(campo); continue; }
    // Guarda o valor sem alterar espaços internos (o encoder usa o texto exato).
    dados[campo] = valor;
  }
  // Tenta converter data e hora somente se os campos forem textos válidos.
  try { if (!invalidos.includes('data')) dados.temporal = lerData(dados.data.trim()); } catch { invalidos.push('data'); }
  // Converte a hora com a mesma regra.
  try { if (!invalidos.includes('hora')) dados.hora_int = lerHora(dados.hora.trim()); } catch { invalidos.push('hora'); }
  // Devolve dados ou a lista de inválidos.
  return invalidos.length ? { invalidos } : { dados };
}

// Transforma os dados validados no vetor de 1.914 colunas, na ordem do ColumnTransformer.
export function montarVetor(dados, avisos = []) {
  // Cria o vetor de saída preenchido com zeros.
  const x = new Float64Array(pesos.camadas[0].W[0].length);
  // Posição inicial do bloco da coluna atual.
  let inicio = 0;
  // Percorre os campos categóricos na ordem do treino.
  for (const [campo, col] of Object.entries(CAMPOS_CATEGORICOS)) {
    // Usa 'missing' para ausência, como o SimpleImputer do treino.
    const valor = dados[campo] === '' || dados[campo].trim() === '' ? preprocesso.valor_ausente : dados[campo];
    // Procura a posição da categoria; tenta também sem espaços extras.
    const pos = INDICES[col].get(valor) ?? INDICES[col].get(valor.trim());
    // Marca 1 na categoria conhecida.
    if (pos !== undefined) x[inicio + pos] = 1;
    // Categoria desconhecida: bloco todo em zero (handle_unknown='ignore') e aviso ao cliente.
    else avisos.push(`Valor de "${campo}" não visto no treino; o campo não contribui para a classificação.`);
    // Avisa quando o campo foi tratado como ausente.
    if (valor === preprocesso.valor_ausente && dados[campo].trim() === '') avisos.push(`"${campo}" não informado; tratado como ausente.`);
    // Avança para o bloco da próxima coluna.
    inicio += preprocesso.categorias[col].length;
  }
  // Reúne os valores numéricos derivados (null = ausente).
  const brutos = { dia_semana: dados.temporal?.dia_semana ?? null, mes: dados.temporal?.mes ?? null, hora: dados.hora_int ?? null };
  // Percorre as colunas numéricas na ordem do treino.
  for (const p of preprocesso.numericas) {
    // Imputa a mediana do treino quando o valor falta.
    const v = brutos[p.nome] ?? p.mediana;
    // Avisa sobre a imputação.
    if (brutos[p.nome] === null) avisos.push(`"${p.nome}" ausente; usada a mediana do treino (${p.mediana}).`);
    // Aplica o StandardScaler: (valor − média) / escala.
    x[inicio++] = (v - p.media) / p.escala;
  }
  // Devolve o vetor pronto para a rede.
  return x;
}

// Executa o forward pass (Linear → ReLU nas ocultas, Linear na saída, sigmoide) e devolve prob. de roubo.
export function forwardPass(x) {
  // Começa com o vetor de entrada.
  let h = x;
  // Percorre todas as camadas lineares.
  pesos.camadas.forEach((camada, i) => {
    // Indica se é a camada de saída (sem ReLU).
    const saida = i === pesos.camadas.length - 1;
    // Calcula cada neurônio: viés + soma(peso × entrada), com ReLU nas ocultas.
    h = camada.W.map((linha, j) => {
      // Inicia a soma com o viés.
      let s = camada.b[j];
      // Acumula o produto escalar.
      for (let k = 0; k < linha.length; k++) s += linha[k] * h[k];
      // Aplica ReLU nas camadas ocultas.
      return saida ? s : Math.max(0, s);
    });
  });
  // Converte o logit em probabilidade com a sigmoide.
  return 1 / (1 + Math.exp(-h[0]));
}

// Handler HTTP chamado pela Netlify a cada requisição.
export default async (req) => {
  // Aceita somente POST.
  if (req.method !== 'POST') return erro(405, 'metodo_nao_permitido', 'Use POST com corpo JSON.');
  // Recusa cedo quando o cabeçalho já declara corpo grande demais.
  if (Number(req.headers.get('content-length') || 0) > LIMITE_BYTES) return erro(413, 'corpo_muito_grande', 'O corpo da requisição excede 8 kB.');
  // Lê o corpo como texto.
  const texto = await req.text();
  // Confere o tamanho real em bytes (o cabeçalho pode faltar).
  if (new TextEncoder().encode(texto).length > LIMITE_BYTES) return erro(413, 'corpo_muito_grande', 'O corpo da requisição excede 8 kB.');
  // Guarda o JSON interpretado.
  let corpo;
  // Tenta interpretar o JSON.
  try { corpo = JSON.parse(texto); } catch { return erro(400, 'json_invalido', 'O corpo precisa ser um JSON válido.'); }
  // Valida campos, tipos, data e hora.
  const { dados, invalidos } = validar(corpo);
  // Devolve 400 com a lista de campos inválidos.
  if (invalidos) return erro(400, 'entrada_invalida', 'Há campos inválidos ou não reconhecidos.', invalidos);
  // Protege a inferência contra falhas inesperadas.
  try {
    // Lista de avisos para o cliente.
    const avisos = [];
    // Calcula a probabilidade de roubo.
    const p = forwardPass(montarVetor(dados, avisos));
    // Responde com a classe estimada e as probabilidades.
    return json(200, {
      // Classe estimada pelo limiar.
      classe: p >= LIMIAR ? 'Roubo' : 'Furto',
      // Probabilidade de roubo com 6 casas.
      prob_roubo: +p.toFixed(6),
      // Probabilidade complementar de furto.
      prob_furto: +(1 - p).toFixed(6),
      // Limiar usado na decisão.
      limiar: LIMIAR,
      // Versão do modelo (hash dos pesos).
      versao_modelo: metadados.versao_modelo,
      // Versão do contrato da API.
      versao_contrato: metadados.versao_contrato,
      // Avisos sobre ausências e categorias desconhecidas.
      avisos,
      // Texto de interpretação obrigatório.
      aviso_interpretacao: AVISO,
    });
  } catch {
    // Responde erro genérico, sem detalhes internos.
    return erro(500, 'erro_interno', 'Não foi possível classificar agora. Tente novamente.');
  }
};
