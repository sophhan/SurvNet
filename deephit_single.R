# Load the torch library
library(torch)

# Set the random seed for reproducibility
torch_manual_seed(123)

# Define hyperparameters
input_size <- 10  # Number of input features (set according to your data)
hidden_units <- c(32,32)  # Number of units in the hidden layer
learning_rate <- 0.001
num_epochs <- 100
dropout_rate <- 0.5  # Dropout probability
cuts <- 9  # Number of cuts for discretization
output_size <- 150 # SHOULD BE IMPLEMENTED AUTOMATICALLY #  phi {torch.tensor} -- Predictions as float tensor with shape [batch, n_durations]
alpha = 0.2
sigma = 0.1

# Generate some dummy survival data for demonstration
batch_size <- 150  # Number of samples/batches
data <- torch_randn(c(batch_size, input_size))
# Dummy survival data (replace with actual survival data)
time <- torch_randint(1, 100, size = c(batch_size, 1))$to(dtype = torch_float())
event <- torch_randint(0, 2, size = c(batch_size, 1))$to(dtype = torch_float())

# Standardize the input data
mean_data <- data$mean(1, keepdim = TRUE)
std_data <- data$std(1, keepdim = TRUE)
data <- (data - mean_data) / std_data
idx_time <- discretize_durations(time, event, cuts = cuts, scheme = "equidistant")

# Define the neural network model
net <- BaseMLP(
  in_features = input_size,
  num_nodes = hidden_units,
  batch_norm = TRUE,
  dropout = dropout_rate,
  activation = "relu",
  out_features = output_size,
  out_bias = TRUE
)

# Define the optimizer (Adam)
optimizer <- optim_adam(net$parameters, lr = learning_rate)

# Training loop with loss tracking
loss_values <- numeric(num_epochs)

# Training loop
for (epoch in 1:num_epochs) {
  # Zero gradients
  optimizer$zero_grad()

  # Forward pass
  predictions <- net(data)
  loss <- deephit_single_loss(predictions, idx_time, event, reduction = "mean")

  # Store loss value
  loss_values[epoch] <- loss$item()

  # Backward pass and optimize
  loss$backward(retain_graph=TRUE)
  optimizer$step()

  # Print loss every 10 epochs
  if (epoch %% 10 == 0) {
    cat("Epoch:", epoch, "Loss:", loss$item(), "\n")
  }
}

# Plot training loss
loss_df <- data.frame(Epoch = 1:num_epochs, Loss = loss_values)

ggplot(loss_df, aes(x = Epoch, y = Loss)) +
  geom_line(color = "blue") +
  labs(title = "Training Loss over Epochs",
       x = "Epoch",
       y = "Loss") +
  theme_minimal()


# deephit single loss
deephit_single_loss <- function(predictions,
                               idx_time,
                               events,
                               reduction = "mean",
                               epsilon = 1e-7,
                               alpha = 0.2,
                               sigma = 0.1) {
  # validate input: Check if the number of prediction outputs is sufficient for the indices in idx_time
  if (ncol(predictions) <= as.numeric(max(idx_time))) {
    stop(
      paste(
        "Network output `predictions` is too small for `idx_time`.",
        "Need at least `predictions.shape[1] =",
        as.numeric(max(idx_time)) + 1,
        ",",
        "but got `predictions.shape[1] =",
        ncol(predictions),
        "`"
      )
    )
  }

  ## Compute negative log likelihood (NLL)
  # flatten events tensor if necessary
  events_row <- torch_flatten(events)

  # reshape idx_time into a 2D column tensor to make it compatible for indexing
  idx_time_col <- torch_reshape(idx_time, c(-1, 1))

  # add a padding column to predictions to handle cases where idx_time includes the last time step
  # padding is added at the end along the second dimension (c(0, 1))
  predictions_pad <- torch::nnf_pad(predictions, c(0, 1), value = 0)  # Padding with zero (or any constant if needed)

  # numerical stability adjustment:
  # extract the maximum value in each row of predictions (along the time dimension)
  # this prevents overflow issues when applying exponential functions later
  gamma <- torch_max(predictions_pad, dim = 2)[[1]]

  # compute the cumulative sum of the exponentials of the normalized predictions
  # normalize by subtracting `gamma` and exponentiating
  cumsum_ <- torch_cumsum(torch_exp(predictions_pad - torch_reshape(gamma, c(-1, 1))), dim = 2)

  # compute the total sum of the exponential terms for each instance (last value in each row of cumsum_)
  sum_ <- cumsum_[, ncol(cumsum_)]

  # part 1: Negative log-likelihood for observed events.
  # extract the predicted value at the observed time indices (idx_time) for each instance
  gathered <- torch_flatten(torch_gather(
    predictions_pad,
    dim = 2,
    index = idx_time_col$to(dtype = torch_int64())
  ))
  # compute the contribution of observed events to the loss
  part1 <- (gathered - gamma) * events_row

  # part 2: Survival probability for all instances
  # compute the log of the total sum of exponentials, ensuring numerical stability using `add(epsilon)`
  part2 <- -sum_$relu()$add(epsilon)$log()

  # part 3: Cumulative probability for censored times
  # compute the remaining cumulative sum at the observed time indices (idx_time)
  surv_remain <- sum_ - torch_flatten(torch_gather(
    cumsum_,
    dim = 2,
    index = idx_time_col$to(dtype = torch_int64())
  ))
  part3 <- (surv_remain$relu()$add(epsilon)$log()) * (1 - events_row)

  # Combine all parts to compute the negative log-likelihood (NLL)
  nll_comp <- -part1$add(part2)$add(part3)

  # apply the reduction method to summarize the rank loss.
  # if reduction = "mean", return the mean of all values.
  # if reduction = "sum", return the sum of all values.
  # otherwise, return the rank loss without any reduction.
  if (reduction == "mean") {
    nll <- nll_comp$mean()
  } else if (reduction == "sum") {
    nll <- nll_comp$sum()
  } else {
    nll <- nll_comp # Return without reduction
  }

  ## Compute rank loss
  # convert predictions to a probability mass function (PMF) using the softmax function.
  # this ensures the predictions are normalized to sum to 1 across the duration axis (dim = 2).
  pmf = predictions_pad$softmax(dim = 2)

  # create a one-hot encoded matrix for observed durations (idx_time)
  # each row corresponds to an observation, with a 1 indicating the observed duration
  y <- torch_zeros_like(pmf)$scatter(2, idx_time_col$to(dtype = torch_int64()), 1)

  # get the number of observations (rows in PMF)
  n <- dim(pmf)[1]

  # create a column vector of ones with the same number of rows as PMF
  ones = torch_ones(c(n, 1), device = pmf$device)

  # compute the cumulative sum of PMF across durations (dim = 2)
  # then calculate a matrix `cdf_diff` that stores the CDF difference for each pair of individuals.
  cdf <- pmf$cumsum(dim = 2)$matmul(y$transpose(1, 2))

  # extract the diagonal of the `cdf` matrix and reshape it into a row vector
  diag_cdf <- torch_reshape(cdf$diag(), c(1, -1))

  # compute the difference in cumulative distribution values for all pairs (i, j).
  # specifically: cdf_diff[i, j] = F_i(T_i) - F_j(T_i)
  cdf_diff <- ones$matmul(diag_cdf) - cdf

  # transpose the matrix to switch rows and columns for easier pairwise comparisons
  cdf_diff <- cdf_diff$transpose(2, 1)

  # reshape the idx_time tensor to ensure it's a flat vector (1D)
  idx_time_row <- torch_reshape(idx_time_col, c(-1))

  # get the total number of durations (n_times)
  n_times <- length(idx_time_row)

  # flatten events tensor if necessary
  events_row <- torch_reshape(events, c(-1))

  # initialize a zero matrix to store the pairwise ranking information
  # shape: [n, n], where n is the number of observations
  mat <- torch_zeros(c(n, n), dtype = torch_float32())

  # compute the rank matrix (`mat`) based on pairwise comparisons.
  # Reshape idx_time_row and events_row for pairwise comparison
  dur_i_matrix <- torch_reshape(idx_time_row, c(-1, 1))
  ev_i_matrix <- torch_reshape(events_row, c(-1, 1))
  # Condition 1: dur_i < dur_j
  condition1 <- dur_i_matrix < idx_time_row
  # Condition 2: dur_i == dur_j and ev_j == 0
  condition2 <- (dur_i_matrix == idx_time_row) & (1 - events_row)
  # Combine conditions into the matrix
  mat <- (condition1 | condition2)$to(dtype = torch_float32())

  # compute the rank loss as specified in the DeepHit paper.
  # multiply the rank matrix by the exponential term involving the CDF difference and sigma.
  rank_loss_prel <- mat * torch_exp(-cdf_diff / sigma)

  # average the rank loss across rows (dim = 2) while keeping dimensions intact
  rank_loss_avg <- rank_loss_prel$mean(dim = 2, keepdim = TRUE)

  # apply the reduction method to summarize the rank loss.
  # if reduction = "mean", return the mean of all values.
  # if reduction = "sum", return the sum of all values.
  # otherwise, return the rank loss without any reduction.
  if (reduction == "mean") {
    rank_loss <- rank_loss_avg$mean()
  } else if (reduction == "sum") {
    rank_loss <- rank_loss_avg$sum()
  } else {
    rank_loss <- rank_loss_avg # Return without reduction
  }

  ## Final loss
  # compute the deephit single loss as specified in the deephit paper.
  return(alpha * nll + (1 - alpha) * rank_loss)
}

