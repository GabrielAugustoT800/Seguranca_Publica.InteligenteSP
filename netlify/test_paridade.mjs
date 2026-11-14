/**
 * Testes da função de classificação. Execução (na raiz do projeto): node netlify/test_paridade.mjs
 * 1) Paridade: compara a probabilidade JS com a do PyTorch em 200 casos da validação (tolerância 1e-4).
 * 2) Contrato: testa entradas válidas (ambas as classes), campos ausentes, categoria desconhecida e erros.
 */

// Importa o handler e as funções internas da própria função de produção.
import classificar, { validar, montarVetor, forwardPass } from './functions/classificar.mjs';
// Importa leitura de arquivos do Node.
import { readFileSync } from 'node:fs';

// Tolerância máxima aceita entre JS e PyTorch.
const TOL = 1e-4;
// Contador de falhas de todos os testes.
let falhas = 0;
// Registra o resultado de uma verificação e conta as falhas.
const checar = (ok, nome) => { console.log(`${ok ? 'OK  ' : 'FALHA'} ${nome}`); if (!ok) falhas++; };

// Lê os casos de referência gravados pelo export_artifacts.py.
const casos = JSON.parse(readFileSync(new URL('./functions/artefatos/casos_teste.json', import.meta.url), 'utf8'));
// Calcula a maior diferença absoluta entre JS e PyTorch.
const difs = casos.map((c) => Math.abs(forwardPass(montarVetor(validar(c.entrada).dados)) - c.prob_roubo_esperada));
// Obtém a diferença máxima.
const max = Math.max(...difs);
// Verifica a paridade em todos os casos.
checar(max <= TOL, `paridade em ${casos.length} casos (dif. máx. ${max.toExponential(2)})`);

// Envia um corpo ao handler e devolve {status, corpo}.
async function chamar(corpo, metodo = 'POST') {
  // Monta a requisição simulada.
  const req = new Request('http://local/api/classificar', { method: metodo, ...(metodo === 'POST' ? { body: typeof corpo === 'string' ? corpo : JSON.stringify(corpo) } : {}) });
  // Executa o handler.
  const res = await classificar(req);
  // Devolve status e JSON.
  return { status: res.status, corpo: await res.json() };
}

// Escolhe o caso de referência com maior probabilidade de roubo.
const casoRoubo = casos.reduce((a, b) => (b.prob_roubo_esperada > a.prob_roubo_esperada ? b : a));
// Escolhe o caso de referência com menor probabilidade de roubo.
const casoFurto = casos.reduce((a, b) => (b.prob_roubo_esperada < a.prob_roubo_esperada ? b : a));
// Testa um caso classificado como Roubo.
const r1 = await chamar(casoRoubo.entrada);
// Verifica status e classe.
checar(r1.status === 200 && r1.corpo.classe === 'Roubo', `classe Roubo (p=${r1.corpo.prob_roubo})`);
// Testa um caso classificado como Furto.
const r2 = await chamar(casoFurto.entrada);
// Verifica status e classe.
checar(r2.status === 200 && r2.corpo.classe === 'Furto', `classe Furto (p=${r2.corpo.prob_roubo})`);
// Testa o corpo vazio (todos os campos ausentes).
const r3 = await chamar({});
// Verifica que classifica com avisos de imputação.
checar(r3.status === 200 && r3.corpo.avisos.length >= 4, 'todos os campos ausentes → 200 com avisos');
// Testa bairro desconhecido.
const r4 = await chamar({ ...casoRoubo.entrada, bairro: 'BAIRRO INEXISTENTE XYZ' });
// Verifica aviso de categoria não vista.
checar(r4.status === 200 && r4.corpo.avisos.some((a) => a.includes('bairro')), 'bairro desconhecido → 200 com aviso');
// Testa data impossível.
checar((await chamar({ data: '2025-02-30' })).status === 400, 'data inválida → 400');
// Testa hora fora do intervalo.
checar((await chamar({ hora: '25:00' })).status === 400, 'hora inválida → 400');
// Testa campo não previsto no contrato.
checar((await chamar({ nome: 'Fulano' })).status === 400, 'campo extra → 400');
// Testa JSON malformado.
checar((await chamar('{ruim')).status === 400, 'JSON inválido → 400');
// Testa corpo acima do limite.
checar((await chamar({ bairro: 'x'.repeat(9000) })).status === 413, 'corpo > 8 kB → 413');
// Testa método GET.
checar((await chamar(null, 'GET')).status === 405, 'GET → 405');

// Encerra com código 1 se houve falha.
console.log(falhas ? `\n${falhas} falha(s).` : '\nTodos os testes passaram.');
// Define o código de saída do processo.
process.exit(falhas ? 1 : 0);
