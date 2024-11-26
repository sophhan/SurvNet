library(survival)

# Function to compute the survival function from the risk scores
predict_outcome <- function(pred_risk_score, time, event, target = "survival") {
  base_hazard <- compute_base_hazard(pred_risk_score, time, event)
  df_risk <- data.frame(
    id = 1:length(as.numeric(predictions)),
    pred_risk_score = as.numeric(predictions)
  )
  df_merged <- merge(df_risk, base_hazard, by = NULL)
  if (target == "hazard") {
    df_out <- df_merged
    df_out$hazard <- df_out$base_hazard * exp(df_out$pred_risk_score)
  } else if (target == "cum_hazard") {
    df_out <- df_merged %>%
      group_by(id) %>%
      arrange(id, -desc(time)) %>%
      mutate(cum_hazard = cumsum(exp(pred_risk_score) * base_hazard))
    df_out <- as.data.frame(df_out)
  } else if (target == "survival") {
    df_out <- df_merged %>%
      group_by(id) %>%
      arrange(id, -desc(time)) %>%
      mutate(survival = exp(-cumsum((exp(pred_risk_score) * base_hazard))))
    df_out <- as.data.frame(df_out)
  } else {
    stop("Invalid target argument. Choose from 'hazard', 'cum_hazard', or 'survival'.")
  }
  return(df_out)
}

# Function to compute baseline hazard
compute_base_hazard <- function(pred_risk_score, time, event) {
  df <- data.frame(
    exp_pred_risk_score = exp(as.numeric(pred_risk_score)),
    time = as.numeric(time),
    event = as.numeric(event)
  )

  df_grouped <- df %>%
    group_by(time) %>%
    summarise(
      total_exp_risk_score = sum(exp_pred_risk_score, na.rm = TRUE),
      total_events = sum(event, na.rm = TRUE)
    ) %>%
    arrange(desc(time)) %>%
    mutate(cum_exp_risk_score = cumsum(total_exp_risk_score),
           base_hazard = total_events/total_exp_risk_score)

  #df_base <- df %>%
  #  left_join(df_grouped %>% select(time, base_hazard), by = "time")

  return(df_grouped[, c("time", "base_hazard")])
}

# Function to discretize durations
discretize_durations <- function(time, event, cuts, scheme = "equidistant") {
  # Convert time and event to numeric
  time <- as.numeric(time)
  event <- as.numeric(event)

  # Initialize an empty vector for the discretized durations
  idx_time <- numeric(length(time))

  # Determine cut scheme and generate cut points accordingly
  if ((length(cuts) == 1) & (scheme == "equidistant")) {
    cuts <- seq(min(time), max(time), length.out = cuts)
  } else if ((length(cuts) == 1) & (scheme == "quantile")) {
    # Fit Kaplan-Meier model
    km_fit <- survfit(Surv(time, event) ~ 1)
    # Extract survival estimates and ordered survival times
    km_est <- km_fit$surv
    km_time <- km_fit$time
    # Generate quantile points from min to max survival probabilities
    s_cuts <- seq(min(km_est), max(km_est), length.out = cuts)
    # Find the closest durations corresponding to these quantiles
    cuts_idx <- sapply(s_cuts, function(q) {
      max(which(km_est >= q))
    })
    # Map indices to duration cut points
    cuts <- km_time[cuts_idx]
    # Remove duplicate cuts to ensure uniqueness
    cuts <- sort(unique(cuts))
    # Set the first cut to min_ or the minimum duration if min_ is NULL
    if (min(time) != min(km_time)) {
      cuts[1] <- min(time)
    }
    # Ensure the last cut is equal to the maximum duration
    if (max(time) != cuts[length(cuts)]) {
      stop("The last cut does not match the maximum duration; something went wrong.")
    }
  } else if ((length(cuts) == 1) & (!scheme %in% c("equidistant", "quantile"))){
    stop("Please provide a valid cut scheme.")
  }

  # Discretize the durations
  idx_time <- sapply(time, function(x) sum(x >= cuts))

  # Return both idx_durations and events
  return(torch_tensor(as.integer(idx_time)))
}


# custom dataset class that prepares the survival data for the dataloader
SurvivalDataset <- dataset(
  initialize = function(data, time, event, normalize = TRUE) {
    # Store raw data
    self$raw_data <- data
    self$raw_time <- time
    self$raw_event <- event

    # Feature normalization (if specified)
    if (normalize) {
      self$mean_data <- data$mean(dim = 1, keepdim = TRUE)
      self$std_data <- data$std(dim = 1, keepdim = TRUE)
      self$data <- (data - self$mean_data) / self$std_data
    } else {
      self$data <- data
    }

    self$time <- time
    self$event <- event
  },

  .getitem = function(index) {
    # Return preprocessed data, time, and event for a specific index
    list(
      data = self$data[index, ],
      time = self$time[index],
      event = self$event[index]
    )
  },

  .length = function() {
    # Return the number of samples in the dataset
    self$data$size(1)
  }
)

