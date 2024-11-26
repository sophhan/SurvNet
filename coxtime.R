############################################################
# Load the torch library
library(torch)

# Set the random seed for reproducibility
torch_manual_seed(123)

# Define hyperparameters
input_features <- 10
input_size <- 11
hidden_units <- c(32,32)
learning_rate <- 0.001
num_epochs <- 100
dropout_rate <- 0.5
output_size <- 1

# Generate some dummy survival data for demonstration
n <- 100  # Number of samples
data <- torch_randn(n, input_features)
# Dummy survival data (replace with actual survival data)
time <- torch_randint(1, 100, size = c(n, 1))$to(dtype = torch_float())
event <- torch_randint(0, 2, size = c(n, 1))$to(dtype = torch_float())
# include time in data
data <- torch_cat(list(data, time), dim = 2)
dim(data)

# Standardize the input data
mean_data <- data$mean(1, keepdim = TRUE)
std_data <- data$std(1, keepdim = TRUE)
data <- (data - mean_data) / std_data

# Define the neural network model
net <- BaseMLP(
  in_features = input_size,
  num_nodes = hidden_units,
  batch_norm = FALSE,
  dropout = dropout_rate,
  activation = "linear",
  out_features = output_size,
  out_bias = FALSE
)

# Define the loss function
# Custom Cox negative log partial likelihood loss function
cox_loss_fn <- function(pred_risk_score, time, event) {
  # Reorder tensors based on sorted indices
  sorted <- torch_argsort(time, dim = 1, descending = FALSE)
  time <- time[sorted]
  event <- event[sorted]
  pred_risk_score <- pred_risk_score[sorted]

  # Compute the partial likelihood
  hazard_ratio <- torch_exp(pred_risk_score)
  log_risk <- torch_log(torch_cumsum(hazard_ratio, dim = 1))
  censored_log_likelihood <- (pred_risk_score - log_risk)*event
  num_observed_events = torch_sum(event)
  neg_log_likelihood <- -torch_sum(censored_log_likelihood)/num_observed_events
  return(neg_log_likelihood)
}

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
  loss <- cox_loss_fn(predictions, time, event)

  # Store loss value
  loss_values[epoch] <- loss$item()

  # Backward pass and optimize
  loss$backward()
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

# Plot survival curves
df_survival <- predict_outcome(predictions, time, event, target = "survival")

ggplot(df_survival, aes(x = time, y = survival, group = id, color = as.factor(id))) +
  geom_line() +
  labs(title = "Survival Function",
       x = "Time",
       y = "Survival Probability") +
  theme_minimal() +
  scale_color_discrete(name = "ID") +
  theme(legend.position = "none")
