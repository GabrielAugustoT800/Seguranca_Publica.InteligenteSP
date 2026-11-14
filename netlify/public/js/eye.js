// Olho digital da tela inicial: a íris acompanha o cursor dentro de um raio limitado.
// Sem mouse (toque ou inatividade), o olho faz uma varredura autônoma suave.

// Seleciona o grupo SVG da íris, que será deslocado.
const iris = document.getElementById('iris');
// Seleciona o elemento SVG do olho, usado para achar o centro na tela.
const olho = document.querySelector('.eye');
// Seleciona o grupo onde serão desenhados os raios da íris.
const raios = document.querySelector('.iris-raios');
// Define o deslocamento máximo da íris em unidades do viewBox (o olho tem raio 200).
const RAIO_MAX = 48;
// Verifica se o usuário pediu menos animação no sistema operacional.
const reduzir = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
// Guarda o deslocamento atual da íris (suavizado).
let atual = { x: 0, y: 0 };
// Guarda o deslocamento-alvo da íris.
let alvo = { x: 0, y: 0 };
// Guarda o instante do último movimento do mouse.
let ultimoMouse = 0;

// Desenha 36 raios finos no núcleo vermelho, imitando a textura de uma íris digital.
for (let i = 0; i < 36; i++) {
  // Calcula o ângulo do raio em radianos.
  const a = (i / 36) * Math.PI * 2;
  // Cria a linha SVG do raio.
  const linha = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  // Define o ponto inicial do raio, perto da pupila.
  linha.setAttribute('x1', (Math.cos(a) * 22).toFixed(2));
  // Define a coordenada Y inicial.
  linha.setAttribute('y1', (Math.sin(a) * 22).toFixed(2));
  // Define o ponto final, com comprimento alternado para dar ritmo.
  linha.setAttribute('x2', (Math.cos(a) * (i % 2 ? 50 : 58)).toFixed(2));
  // Define a coordenada Y final.
  linha.setAttribute('y2', (Math.sin(a) * (i % 2 ? 50 : 58)).toFixed(2));
  // Acrescenta o raio ao grupo da íris.
  raios.appendChild(linha);
}

// Converte a posição do cursor em um alvo de deslocamento da íris.
function mirar(clientX, clientY) {
  // Obtém a posição e o tamanho do olho na tela.
  const r = olho.getBoundingClientRect();
  // Calcula o vetor do centro do olho até o cursor, em pixels.
  const dx = clientX - (r.left + r.width / 2);
  // Calcula a componente vertical do vetor.
  const dy = clientY - (r.top + r.height / 2);
  // Calcula a distância até o cursor.
  const dist = Math.hypot(dx, dy) || 1;
  // Limita o deslocamento: cresce com a distância e satura em RAIO_MAX.
  const k = Math.min(RAIO_MAX, dist / 6) / dist;
  // Define o alvo na mesma direção do cursor.
  alvo = { x: dx * k, y: dy * k };
}

// Atualiza o alvo sempre que o mouse se move.
window.addEventListener('mousemove', (e) => {
  // Registra o instante do movimento.
  ultimoMouse = performance.now();
  // Recalcula o alvo da íris.
  mirar(e.clientX, e.clientY);
});

// Volta a íris ao centro quando o cursor sai da janela.
document.addEventListener('mouseleave', () => { alvo = { x: 0, y: 0 }; });

// Laço de animação: aproxima a íris do alvo e cria a varredura autônoma.
function quadro(t) {
  // Sem mouse há mais de 3 s, o olho "procura" sozinho seguindo uma curva de Lissajous.
  if (!reduzir && t - ultimoMouse > 3000) alvo = { x: Math.sin(t / 1700) * RAIO_MAX * 0.7, y: Math.sin(t / 2300) * RAIO_MAX * 0.4 };
  // Suaviza o movimento (interpolação de 12% por quadro).
  atual.x += (alvo.x - atual.x) * (reduzir ? 1 : 0.12);
  // Suaviza a componente vertical.
  atual.y += (alvo.y - atual.y) * (reduzir ? 1 : 0.12);
  // Aplica o deslocamento ao grupo da íris.
  iris.setAttribute('transform', `translate(${atual.x.toFixed(2)} ${atual.y.toFixed(2)})`);
  // Agenda o próximo quadro.
  requestAnimationFrame(quadro);
}

// Inicia a animação.
requestAnimationFrame(quadro);
