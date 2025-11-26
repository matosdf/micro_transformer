import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import time
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. CONFIGURAÇÃO DE AMBIENTE E HARDWARE
# ==========================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"--- Configuração de Hardware ---")
if device == 'cuda':
    print(f"Dispositivo: GPU ({torch.cuda.get_device_name(0)})")
else:
    print("Dispositivo: CPU")

# ==========================================
# 2. HIPERPARÂMETROS
# ==========================================
batch_size = 64
block_size = 256
max_iters = 15000
eval_interval = 250
learning_rate = 1e-3
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
target_loss = 1.29  # Meta de parada

torch.manual_seed(1337)

# ==========================================
# 3. CARREGAMENTO DE DADOS
# ==========================================
# [R: Carregar Dados]
# Código equivalente à leitura de arquivo e criação do vocabulário em R.
# Python usa 'open()' e 'set()' nativos em vez de 'readr' e 'stringr'.
file_path = 'machado.txt'

if not os.path.exists(file_path):
    print(f"\nArquivo '{file_path}' não encontrado. Baixando...")
    url = "https://www.gutenberg.org/cache/epub/55752/pg55752.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        text_content = response.text
        start_idx = text_content.find("*** START OF THE PROJECT GUTENBERG EBOOK")
        end_idx = text_content.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        if start_idx != -1 and end_idx != -1:
            text_content = text_content[start_idx:end_idx]
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
    except Exception as e:
        print(f"Erro ao baixar: {e}. Usando texto dummy.")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("Exemplo de texto dummy. " * 5000)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulário: {vocab_size} caracteres")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]


# [R: get_batch]
# Função para criar lotes (batches) de dados de entrada e alvo.
# Diferença: R usa 'outer' para criar a matriz de índices; Python usa 'torch.stack'
# para empilhar as sequências cortadas.
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# ==========================================
# 4. ARQUITETURA DO MODELO (GPT)
# ==========================================

# [R: Head]
# Implementa uma única cabeça de auto-atenção (Self-Attention).
# Diferença: R usa 'nn_buffer' para a máscara triangular; Python usa 'register_buffer'.
# A lógica de Query, Key, Value e Masked Softmax é idêntica.
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


# [R: MultiHeadAttention]
# Gerencia múltiplas cabeças de atenção em paralelo.
# Diferença: R usa 'lapply' para iterar sobre as cabeças; Python usa List Comprehension
# dentro de um 'nn.ModuleList'. O resultado é concatenado (torch.cat) e projetado.
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# [R: FeedForward]
# Camada densa simples (MLP) com não-linearidade (ReLU).
# A estrutura é idêntica: Linear (4x) -> ReLU -> Linear (proj).
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# [R: Block]
# Bloco principal do Transformer: Comunicação (Atenção) + Computação (FeedForward).
# Implementa a arquitetura "Pre-Norm" (normalização antes das operações), exatamente como no R.
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# [R: GPT / BaseModel]
# Modelo completo combinando Embeddings, Positional Encoding e Blocos Sequenciais.
# A classe 'BaseModel' do R foi absorvida aqui.
# Define a estrutura global do Transformer Decoder-Only.
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > block_size:
            idx = idx[:, -block_size:]
            T = block_size

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # [R: generate]
    # Gera novos tokens iterativamente a partir de um contexto.
    # Lógica: Cortar contexto -> Forward -> Softmax -> Sample -> Concatenar.
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# [R: estimate_loss]
# Calcula a perda média nos conjuntos de treino e validação sem atualizar gradientes.
# Equivalente ao uso de 'local_no_grad()' no R.
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ==========================================
# 5. EXECUÇÃO COM GRÁFICOS E PARADA
# ==========================================

model = GPT()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Listas para armazenar histórico para o gráfico
train_loss_history = []
val_loss_history = []
step_history = []

print(f"\n--- Iniciando Treinamento ---")
print(f"Meta de Loss (Validação): < {target_loss}")

start_time = time.time()
pbar = tqdm(range(max_iters), desc="Treinando")

# [R: treinar]
# Loop principal de otimização.
# Diferença: Python usa 'optimizer.zero_grad' / 'step' explicitamente.
# Adicionada aqui a lógica de parada antecipada (Early Stopping) e coleta de histórico.
for iter in pbar:

    # Avaliação periódica
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)

        # Armazenar histórico
        train_loss_history.append(losses['train'])
        val_loss_history.append(losses['val'])
        step_history.append(iter)

        # Atualizar descrição da barra de progresso
        pbar.set_description(f"Step {iter} | Train: {losses['train']:.3f} Val: {losses['val']:.3f}")

        # 1. CONDIÇÃO DE PARADA (STOPPING CONDITION)
        if losses['val'] < target_loss:
            print(f"\n\n[SUCESSO] Meta atingida! Val Loss {losses['val']:.4f} < {target_loss}")
            print(f"Parando treinamento no passo {iter}.")
            break

    # Treinamento padrão
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = time.time()
print(f"\nTreinamento finalizado em {end_time - start_time:.2f} segundos.")

# ==========================================
# 6. GERAR GRÁFICO DE LOSS
# ==========================================
print("\n--- Gerando Gráfico de Desempenho ---")

plt.figure(figsize=(10, 6))
plt.plot(step_history, train_loss_history, label='Treino Loss', color='blue')
plt.plot(step_history, val_loss_history, label='Validação Loss', color='orange')
plt.axhline(y=1.3, color='red', linestyle='--', label=f'Meta 1.3')

plt.title('Evolução da Loss durante o Treinamento')
plt.xlabel('Iterações (Steps)')
plt.ylabel('Loss (Cross-Entropy)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 7. GERAÇÃO DE TEXTO
# ==========================================
print("\n--- Gerando Texto ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = model.generate(context, max_new_tokens=1000)
print(decode(generated_indices[0].tolist()))