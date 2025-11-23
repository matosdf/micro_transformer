
dev.off(dev.list()["RStudioGD"])  
rm(list=ls())                       
cat("\f")

# CARREGAR OS DADOS E GERAR A FUNÇÃO GETBATCH    ----

library(torch)
library(dotty)


data <- readr::read_file("D://MAURI/2025/CURSOS/PO_249___2025/AULA___11/machado.txt")

stringr::str_sub(data, 100, 1000)

vocab <- sort(unique(stringr::str_split_1(data, "")))
vocab <- structure(seq_along(vocab), names = vocab)

data <- stringr::str_split_1(data, "")


get_batch <- function(split, block_size = 8, batch_size = 4) {
  n <- length(data)
  
  batch_ids <- if (split == "train") {
    sample.int(0.8*(n - block_size), batch_size)
  } else {
    sample(seq(as.integer(0.8*n) + block_size, n - block_size*2, 1L), size= 1L) 
  }
  
  batch_ids <- outer(batch_ids, 0:(block_size-1), "+")
  
  x <- vocab[data[batch_ids]]
  dim(x) <- dim(batch_ids)
  
  y <- vocab[data[batch_ids + 1]]
  dim(y) <- dim(batch_ids)
  
  list(x = torch_tensor(x), y = torch_tensor(y)) 
}

get_batch("train")



# MODELO BASE    ----

BaseModel <- nn_module(
  initialize = function(vocab_size) {
    self$vocab_size <- vocab_size
    self$token_embeddings_table <- nn_embedding(vocab_size, vocab_size)
  },
  forward = function(x, targets = NULL) {
    logits <- self$token_embeddings_table(x) 
    
    if (is.null(targets)) {
      return(logits)
    }
    
    .[b, t, c] <- logits$shape
    logits <- logits$view(c(b*t, c))
    targets <- targets$view(c(b*t))
    loss <- nnf_cross_entropy(logits, targets)
    
    list(logits, loss)
  }
)

m <- BaseModel(length(vocab))
.[xb, yb] <- get_batch("train")
.[logits, loss] <- m(xb, yb)



# FUNÇÃO GENERATE    ----

generate <- function(m, texto = "'", max_len=100) {
  local_no_grad()
  
  for (i in seq_len(max_len)) {
    x <- torch_tensor(vocab[texto])$view(c(1, -1))
    logits <- m(x) 
    logits <- logits[,-1,] 
    probs <- as.numeric(nnf_softmax(logits, dim = -1))
    prox <- sample(names(vocab), size = 1, prob = probs)
    texto <- c(texto, prox)
  }
  
  paste(texto, collapse="")
}



# FUNÇÃO ESTIMATE LOSS    ----

estimate_loss <- function(m, steps = 100, block_size = 8) {
  local_no_grad() 
  m$eval()
  losses <- list(train = numeric(steps), valid = numeric(steps))
  for (mode in c("train", "valid")) {
    for (i in 1:100) {
      .[xb, yb] <- get_batch(mode, block_size, 32)
      .[logits, loss] <- m(xb, yb)
      losses[[mode]][i] <- loss$item()
    }
  }
  m$train()
  lapply(losses, mean)
}



# FUNÇÃO TREINAR    ----

treinar <- function(m, steps = 5000, block_size = 8, batch_size = 32) {
  
  device <- if (cuda_is_available()) {
    "cuda"
  } else "cpu"
  
  m$to(device=device)
  opt <- optim_adamw(m$parameters)
  
  losses <- list()
  for (i in seq_len(steps)) {
    
    .[xb, yb] <- get_batch("train", block_size, batch_size)
    .[logits, loss] <- m(xb$to(device=device), yb$to(device=device))
    
    opt$zero_grad()
    loss$backward()
    opt$step()
    
    if (i %% (steps / 10) == 0) {
      .[train_loss, valid_loss] <- estimate_loss(m, block_size = block_size)
      cat(glue::glue("iter: {i} - train_loss: {train_loss} - valid_loss: {valid_loss}"), "\n")
    }
  }
  
  m$to(device="cpu")
  invisible(NULL)
}



# HEAD    ----

head_size <- 16



Head <- nn_module(
  initialize = function(n_embd, head_size, block_size) {
    self$query <- nn_linear(n_embd, head_size, bias = FALSE)
    self$key <- nn_linear(n_embd, head_size, bias = FALSE)
    self$value <- nn_linear(n_embd, head_size, bias = FALSE)
    
    self$tril <- nn_buffer(torch_tril(torch_ones(block_size, block_size)))
    self$dropout <- nn_dropout(0.2)
  },
  forward = function(x) {
    
    .[b, t, c] <- x$shape
    
    k <- self$key(x)
    q <- self$query(x)
    v <- self$value(x)
    
    wei <- torch_matmul(k, q$transpose(-2, -1)) / sqrt(c)
    wei$masked_fill_(self$tril[1:t, 1:t] == 0, -Inf)
    wei <- nnf_softmax(wei, dim = -1)
    wei <- self$dropout(wei)
    
    out <- torch_matmul(wei, v)
    out
  }
)



# MULTI HEAD    ----

MultiHeadAttention <- nn_module(
  initialize = function(n_head, n_embd, head_size, block_size) {
    self$heads <- nn_module_list(
      replicate(n_head, Head(n_embd, head_size, block_size))
    )
    self$proj <- nn_linear(n_embd, n_embd)
  },
  forward = function(x) {
    self$heads |> 
      lapply(function(h) h(x)) |> 
      torch_cat(dim = -1) |> 
      self$proj()
  }
)




# FEED FORWARD    ----

FeedForward <- nn_module(
  initialize = function(n_embd) {
    self$linear <- nn_linear(n_embd, n_embd * 4)
    self$proj <- nn_linear(4* n_embd, n_embd)
  },
  forward = function(x) {
    x |> 
      self$linear() |> 
      nnf_relu() |> 
      self$proj()
  }
)



# BLOCK    ----

Block <- nn_module(
  initialize = function(n_head, n_embd, block_size) {
    self$sa_head <- MultiHeadAttention(n_head, n_embd, n_embd/n_head, block_size)
    self$ffwd <- FeedForward(n_embd)
    
    self$ln1 <- nn_layer_norm(n_embd)
    self$ln2 <- nn_layer_norm(n_embd)
    
    self$dropout <- nn_dropout(0.2)
  },
  forward = function(x) {
    x <- x + self$dropout(self$sa_head(self$ln1(x))) 
    x <- x + self$dropout(self$ffwd(self$ln2(x)))
    x
  }
)


# GPT    ----

GPT <- nn_module(
  initialize = function(vocab_size, n_block = 3, n_embd = 32, block_size = 8, n_head = 4) {
    self$vocab_size <- vocab_size
    self$block_size <- block_size
    
    self$token_embeddings_table <- nn_embedding(vocab_size, n_embd)
    self$pos_embedding_table <- nn_embedding(block_size, n_embd)
    self$blocks <- nn_sequential(!!!replicate(n_block, Block(n_head, n_embd, block_size)))
    self$ln <- nn_layer_norm(n_embd)
    self$lm_head <- nn_linear(n_embd, vocab_size)
  },
  forward = function(x, targets = NULL) {
    
    x <- x[,-self$block_size:N]
    .[b, t] <- x$shape
    
    tok_emb <- self$token_embeddings_table(x) 
    pos_emb <- self$pos_embedding_table(torch_arange(1, t, dtype="int")) 
    
    x <- tok_emb + pos_emb 
    x <- self$blocks(x)
    x <- self$ln(x)
    
    logits <- self$lm_head(x) 
    
    if (is.null(targets)) {
      return(logits)
    }
    
    .[b, t, c] <- logits$shape
    logits <- logits$view(c(b*t, c))
    targets <- targets$view(c(b*t))
    loss <- nnf_cross_entropy(logits, targets)
    
    list(logits, loss)
  }
)



# RODAR O MODELO MAIS SIMPLES    ----

m <- GPT(vocab_size = length(vocab), block_size = 8)

system.time(
            treinar(m, steps = 500)
)

cat(generate(m))


#********************************************    ----

# RODAR O MODELO MAIS COMPLETO    ----

m <- GPT(
  vocab_size = length(vocab), 
  block_size = 256,
  n_block = 6, 
  n_head = 6,
  n_embd = 384
)

system.time(
  treinar(m, steps = 10000, block_size = 256, batch_size = 64)
            )



  












