library(readr)
library(stringr)
library(torch)
library(dotty)

data <- readr::read_file("C:/Users/matos/OneDrive/Área de Trabalho/Programacao/micro_transformer/machado.txt")

stringr::str_sub(data, 100, 1000)

vocab <- sort(unique(stringr::str_split_1(data, "")))
vocab <- structure(seq_along(vocab), names = vocab)

stringr::str_length(data)

data <- stringr::str_split_1(data, "")
length(data)
data[1:18]


get_batch <- function(split, block_size = 8, batch_size = 4) {
  n <- length(data)
  batch_ids <- if (split == "train") {
    sample.int(0.8*(n - block_size), batch_size)
  } else {
    sample(seq(as.integer(0.8*n) + block_size, n - block_size*2, 1L),
    size= 1L)
  }
  batch_ids <- outer(batch_ids, 0:(block_size-1), "+")
  
  x <- vocab[data[batch_ids]]
  dim(x) <- dim(batch_ids)
  
  y <- vocab[data[batch_ids + 1]]
  dim(y) <- dim(batch_ids)
  
  list(x = torch_tensor(x), y = torch_tensor(y))
}

get_batch("train")
.[xb, yb] <- get_batch("train")
xb
yb
    
  