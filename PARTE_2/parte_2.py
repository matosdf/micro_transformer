import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import time

# ==============================================================================
# CONFIGURAÇÕES E HIPERPARÂMETROS
# ==============================================================================
# Garante reprodutibilidade
torch.manual_seed(1337)

# Hiperparâmetros (baseados no código R e PDF)
BATCH_SIZE = 64  # R: batch_size = 64
BLOCK_SIZE = 8  # R: block_size = 8
MAX_ITERS = 10000  # R: steps (na função treinar, ajustado para teste rápido)
LEARNING_RATE = 1e-3  # R: lr = 0.001
EVAL_INTERVAL = 1000  # A cada quanto tempo verificamos a loss
EVAL_ITERS = 100  # Quantas iterações para estimar a média da loss
N_EMBD = 32  # R: n_embd (tamanho do vetor de embedding)

# Configuração de dispositivo (CPU ou GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Executando no dispositivo: {device}")

# Caminho do arquivo
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(script_dir, '..', 'machado.txt')


# ==============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO (Vindo da Parte I)
# ==============================================================================
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("Arquivo não encontrado. Usando texto dummy.")
        text = "Aqui está um texto de exemplo para simular o funcionamento." * 500

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    return data, vocab_size, encode, decode


data, vocab_size, encode, decode = load_data(FILE_PATH)
n = len(data)
train_data = data[:int(n * 0.8)]
val_data = data[int(n * 0.8):]


def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data_source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    # Importante: mover os dados para o dispositivo (GPU/CPU)
    x, y = x.to(device), y.to(device)
    return x, y


# ==============================================================================
# 2. FUNÇÃO ESTIMAR LOSS (R: estimar_loss)
# ==============================================================================
@torch.no_grad()  # R: local_no_grad()
def estimate_loss(model):
    out = {}
    model.eval()  # R: modelo$eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # R: modelo$train()
    return out


# ==============================================================================
# 3. CAMADA DE EMBEDDINGS MÉDIOS (R: AverageEmbeddings)
# ==============================================================================
class AverageEmbeddings(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        # R: self$wei <- nn_parameter(torch_randn(block_size, block_size))
        # Nota: O código R usa pesos aprendíveis aleatórios para a média.
        self.wei = nn.Parameter(torch.randn(block_size, block_size))

    def forward(self, x):
        # x shape: (B, T, C)
        B, T, C = x.shape

        # Cria a matriz triangular inferior (Mask)
        # R: tril <- torch_tril(torch_ones(t, t))
        tril = torch.tril(torch.ones(T, T, device=device))

        # Aplica a máscara: onde é 0 vira -Inf (para o softmax zerar depois)
        # R: wei <- self$wei[1:t, 1:t]$masked_fill(tril == 0, -Inf)
        wei = self.wei[:T, :T]
        wei = wei.masked_fill(tril == 0, float('-inf'))

        # Softmax para "normalizar" os pesos
        # R: wei <- nnf_softmax(wei, dim = -1)
        wei = F.softmax(wei, dim=-1)

        # Multiplicação de matrizes (A agregação em si)
        # R: torch_matmul(wei, x)
        out = wei @ x
        return out


# ==============================================================================
# 4. MODELO GPT INICIAL (R: GPT)
# ==============================================================================
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=N_EMBD, block_size=BLOCK_SIZE):
        super().__init__()
        # Tabela de Embeddings (Token -> Vetor)
        # R: self$token_embedding_layer <- nn_embedding(vocab_size, n_embd)
        self.token_embedding_layer = nn.Embedding(vocab_size, n_embd)

        # Camada de média (que definimos acima)
        # R: self$average_embeddings <- AverageEmbeddings()
        self.average_embeddings = AverageEmbeddings(block_size)

        # Cabeça Linear final (Vetor -> Logits/Probabilidades)
        # R: self$lm_head <- nn_linear(n_embd, vocab_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 1. Embeddings
        # R: emb <- self$token_embedding_layer(x)
        emb = self.token_embedding_layer(idx)  # (B, T, n_embd)

        # 2. Média (Atenção simplificada)
        # R: avg <- self$average_embeddings(emb)
        # Nota: No código da Parte II em R, 'avg' é calculado mas NÃO é usado
        # na linha seguinte para gerar logits. Estamos replicando esse comportamento
        # para fidelidade à prova, embora num Transformer real usaríamos 'avg'.
        avg = self.average_embeddings(emb)

        # 3. Logits
        # R: logits <- self$lm_head(emb)  <-- aqui o código em R usa 'emb', não 'avg'
        logits = self.lm_head(emb)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cálculo do NLL (Negative Log Likelihood / Cross Entropy)
            # R: calcular_nll(logits, yb) ... nnf_cross_entropy(...)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # R: generate <- function(...)
    def generate(self, idx, max_new_tokens):
        # idx é (B, T) array de indices no contexto atual
        for _ in range(max_new_tokens):
            # Corta o contexto se for maior que o block_size (necessário p/ AverageEmbeddings)
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Pega as previsões
            logits, _ = self(idx_cond)

            # Foca apenas no último passo de tempo (o último token gerado)
            # R: logits <- logits[,-1,]
            logits = logits[:, -1, :]  # Fica (B, C)

            # Aplica softmax para ter probabilidades
            # R: probs <- as.numeric(nnf_softmax(logits, dim = -1))
            probs = F.softmax(logits, dim=-1)

            # Amostra da distribuição (sample)
            # R: prox <- sample(names(vocab), size = 1, prob = probs)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Concatena o novo token à sequência
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==============================================================================
# 5. TREINAMENTO (R: treinar e bloco principal)
# ==============================================================================
if __name__ == "__main__":
    # Instancia o modelo
    model = GPT(vocab_size, N_EMBD, BLOCK_SIZE)
    model = model.to(device)

    # Define o otimizador
    # R: opt <- optim_adamw(modelo$parameters, lr = 0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Iniciando treinamento por {MAX_ITERS} iterações...")

    # --- INÍCIO DA MEDIÇÃO DE TEMPO ---
    start_time = time.time()

    # Loop de Treino
    for iter in range(MAX_ITERS):

        # A cada X passos, avalia a loss nos dados de treino e validação
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Pega um batch
        xb, yb = get_batch('train')

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass (Backpropagation)
        optimizer.zero_grad(set_to_none=True)  # R: opt$zero_grad()
        loss.backward()  # R: nll$backward()
        optimizer.step()  # R: opt$step()

    # --- FIM DA MEDIÇÃO DE TEMPO ---
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Treinamento finalizado.")
    print(f"Tempo de processamento: {elapsed_time:.4f} segundos")

    # ==========================================================================
    # 6. GERAÇÃO DE TEXTO (R: generate(m))
    # ==========================================================================
    print("\n--- Gerando Texto ---")
    # Contexto inicial vazio (índice 0 ou quebra de linha)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Gera 100 novos tokens
    generated_ids = model.generate(context, max_new_tokens=100)

    # Decodifica para texto
    print(decode(generated_ids[0].tolist()))