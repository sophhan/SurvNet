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
batch_size <- 100

# Generate some dummy survival data for demonstration
n <- 10000  # Number of samples
data <- torch_randn(n, input_features)
# Dummy survival data (replace with actual survival data)
time <- torch_randint(1, 100, size = c(n, 1))$to(dtype = torch_float())
event <- torch_randint(0, 2, size = c(n, 1))$to(dtype = torch_float())
# include time in data
data <- torch_cat(list(data, time), dim = 2)
dim(data)

# Split the data into training and validation sets
train_size <- 0.8 * n  # 80% training, 20% validation
train_indices <- sample(1:n, train_size)
valid_indices <- setdiff(1:n, train_indices)

train_data <- data[train_indices, ]
train_time <- time[train_indices, ]
train_event <- event[train_indices, ]

valid_data <- data[valid_indices, ]
valid_time <- time[valid_indices, ]
valid_event <- event[valid_indices, ]

# Create survival datasets
train_dataset <- SurvivalDataset(train_data, train_time, train_event, normalize = TRUE)
valid_dataset <- SurvivalDataset(valid_data, valid_time, valid_event, normalize = TRUE)

# Create dataloaders
train_dataloader <- dataloader(
  dataset = train_dataset,
  batch_size = batch_size,
  shuffle = TRUE
)

valid_dataloader <- dataloader(
  dataset = valid_dataset,
  batch_size = batch_size,
  shuffle = FALSE  # No shuffling for validation
)

valid_dataset$data
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
training_loss <- numeric(num_epochs)
validation_loss <- numeric(num_epochs)

# Training loop with validation
# Training loop with validation
for (epoch in 1:num_epochs) {
  # Training Phase
  net$train()  # Set the network to training mode
  total_train_loss <- 0

  coro::loop(for (batch in train_dataloader) {
    batch_data <- batch$data
    batch_time <- batch$time
    batch_event <- batch$event

    optimizer$zero_grad()
    predictions <- net(batch_data)
    train_loss <- cox_loss_fn(predictions, batch_time, batch_event)
    total_train_loss <- total_train_loss + train_loss$item()

    train_loss$backward()
    optimizer$step()
  })

  avg_train_loss <- total_train_loss / length(train_dataloader)

  # Validation Phase
  net$eval()  # Set the network to evaluation mode (disables dropout, etc.)
  total_valid_loss <- 0

  with_no_grad({
    coro::loop(for (batch in valid_dataloader) {
      batch_data <- batch$data
      batch_time <- batch$time
      batch_event <- batch$event

      predictions <- net(batch_data)
      valid_loss <- cox_loss_fn(predictions, batch_time, batch_event)
      total_valid_loss <- total_valid_loss + valid_loss$item()
    })
  })

  avg_valid_loss <- total_valid_loss / length(valid_dataloader)

  # Store training and validation losses
  training_loss[epoch] <- avg_train_loss
  validation_loss[epoch] <- avg_valid_loss

  # Print losses every 10 epochs
  if (epoch %% 10 == 0) {
    cat("Epoch:", epoch,
        "Avg Train Loss:", avg_train_loss,
        "Avg Valid Loss:", avg_valid_loss, "\n")
  }
}

# Plot training loss
loss_df <- data.frame(
  Epoch = 1:num_epochs,
  TrainLoss = training_loss,
  ValidLoss = validation_loss
)

ggplot(loss_df) +
  geom_line(aes(x = Epoch, y = TrainLoss, color = "Train")) +
  geom_line(aes(x = Epoch, y = ValidLoss, color = "Validation")) +
  labs(title = "Training and Validation Loss over Epochs",
       x = "Epoch",
       y = "Loss",
       color = "Dataset") +
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
