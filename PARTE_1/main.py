import torch
import os

# ==============================================================================
# CONFIGURAÇÕES INICIAIS
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))  # Pega o caminho completo de PARTE_1
FILE_PATH = os.path.join(script_dir, '..', 'machado.txt')  # Sobe um nível e acha machado.txt

BLOCK_SIZE = 8  # Tamanho do contexto (janela de tempo)
BATCH_SIZE = 4  # Quantas sequências processar em paralelo
TRAIN_SPLIT = 0.8  # 80% para treino, 20% para validação

# ==============================================================================
# 1. CARREGAMENTO DOS DADOS E VOCABULÁRIO
# ==============================================================================
def load_data(file_path):
    """
    Lê o arquivo, cria o vocabulário e converte o texto para tensores inteiros.
    Equivalente ao trecho em R que usa readr::read_file e stringr::str_split.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Arquivo '{file_path}' carregado com sucesso.")
        # Equivalente ao stringr::str_sub(data, 100, 1000) apenas para visualização
        print(f"Amostra do texto: {text[100:200]}...")
    except FileNotFoundError:
        print(f"Arquivo '{file_path}' não encontrado. Usando texto de exemplo.")
        text = "Aqui está um texto de exemplo para simular o funcionamento do código. " * 500

    # --------------------------------------------------------------------------
    # Construção do Vocabulário
    # R: vocab <- sort(unique(stringr::str_split_1(data, "")))
    # --------------------------------------------------------------------------
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Tamanho do vocabulário: {vocab_size}")

    # --------------------------------------------------------------------------
    # Mapeamentos (Encoder/Decoder)
    # R: vocab <- structure(seq_along(vocab), names = vocab) (cria índices nomeados)
    # Python: Criamos dicionários explícitos para conversão char <-> int
    # --------------------------------------------------------------------------
    stoi = {ch: i for i, ch in enumerate(chars)}  # String to Int
    itos = {i: ch for i, ch in enumerate(chars)}  # Int to String

    # Função para codificar: string -> lista de inteiros
    encode = lambda s: [stoi[c] for c in s]

    # Função para decodificar: lista de inteiros -> string
    decode = lambda l: ''.join([itos[i] for i in l])

    # --------------------------------------------------------------------------
    # Criação do Tensor de Dados
    # R: data <- stringr::str_split_1(data, "") ... depois acessa via índices
    # Python: Convertemos o texto para um tensor LongTensor (inteiros)
    # --------------------------------------------------------------------------
    data_tensor = torch.tensor(encode(text), dtype=torch.long)
    print(f"Tamanho total do dataset (caracteres): {len(data_tensor)}")
    print(f"Primeiros 18 tokens (caracteres): '{decode(data_tensor[:18].tolist())}'")

    return data_tensor, vocab_size, encode, decode


# ==============================================================================
# 2. SEPARAÇÃO TREINO / VALIDAÇÃO
# ==============================================================================
# Carrega os dados
data, vocab_size, encode, decode = load_data(FILE_PATH)

# Define o ponto de corte (n no R refere-se ao length(data))
n = len(data)
n_train = int(n * TRAIN_SPLIT)

# Divide os dados em dois tensores distintos
train_data = data[:n_train]
val_data = data[n_train:]


# ==============================================================================
# 3. FUNÇÃO GET_BATCH
# ==============================================================================
def get_batch(split, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    """
    Gera um lote (batch) de entradas (x) e alvos (y).
    Args:
        split (str): 'train' ou 'valid' (qualquer coisa diferente de train).
        block_size (int): Tamanho do contexto.
        batch_size (int): Número de sequências no batch.

    Returns:
        x (torch.Tensor): Tensor de entrada (batch_size, block_size)
        y (torch.Tensor): Tensor alvo (batch_size, block_size)
    """
    # Seleciona o dataset correto
    # R: if (split == "train") { sample... } else { sample... }
    data_source = train_data if split == 'train' else val_data

    # Gera índices aleatórios para o início de cada bloco
    # R: sample.int(0.8*(n-block_size), batch_size)
    # Python: torch.randint gera inteiros aleatórios até high (len(data_source) - block_size)
    ix = torch.randint(len(data_source) - block_size, (batch_size,))

    # Constrói o tensor X empilhando os pedaços do texto
    # R: x <- vocab[data[batch_ids]]; dim(x) <- dim(batch_ids)
    # Python: Para cada índice i em ix, pegamos o slice data[i : i+block_size]
    x = torch.stack([data_source[i: i + block_size] for i in ix])

    # Constrói o tensor Y (alvo) deslocado em 1 posição à direita
    # R: y <- vocab[data[batch_ids + 1]]
    y = torch.stack([data_source[i + 1: i + block_size + 1] for i in ix])

    # R: list(x = torch_tensor(x), y = torch_tensor(y))
    # Python: No PyTorch, x e y já são tensores aqui se data_source for tensor.
    # Mas garantimos que estejam no dispositivo correto (CPU/GPU) se necessário no futuro.
    return x, y


# ==============================================================================
# TESTE DA IMPLEMENTAÇÃO (Main)
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(1337)  # Para reprodutibilidade, assim como set.seed no R

    print("\n--- Testando get_batch('train') ---")
    xb, yb = get_batch('train')

    print("Shape de X:", xb.shape)  # Deve ser [4, 8] (batch_size, block_size)
    print("Shape de Y:", yb.shape)

    print("\nConteúdo de X (Inputs):")
    print(xb)

    print("\nConteúdo de Y (Targets/Labels):")
    print(yb)

    print("\n--- Verificação Visual (Decodificação) ---")
    # Vamos pegar a primeira linha do batch para conferir se Y é o deslocamento de X
    primeira_linha_x = xb[0].tolist()
    primeira_linha_y = yb[0].tolist()

    print(f"X[0] decodificado: '{decode(primeira_linha_x)}'")
    print(f"Y[0] decodificado: '{decode(primeira_linha_y)}'")
    print("Nota: Y deve ser X deslocado uma letra para frente (next token prediction).")