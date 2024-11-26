DeepSurv <- torch::nn_module(
  classname = "DeepSurv",
  initialize = function(net, base_hazard,
                        preprocess_fun = NULL,
                        postprocess_fun = NULL) {
    self$net <- net
    self$base_hazard <- base_hazard
    self$t <- torch::torch_tensor(base_hazard$time)
    self$t_orig <- self$base_hazard$time

    if (is.null(preprocess_fun)) {
      self$preprocess_fun <- function(x) {
        if (is.list(x) && length(x) == 1) {
          x <- x[[1]]
        } else if (is.list(x)) {
          stop("Currently, only one input layer is supported!")
        }

        x
      }
    } else {
      self$preprocess_fun <- preprocess_fun
    }

    if (is.null(postprocess_fun)) {
      self$postprocess_fun <- function(x) {
        x <- x$unsqueeze(-1) * torch_ones(c(dim(x), length(self$t)))
        x$unsqueeze(-2)
      }
    } else {
      self$postprocess_fun <- postprocess_fun
    }
  },

  # Input is a list of lists of (features, time)
  forward = function(input, target = "survival", use_base_hazard = TRUE) {

    # Remove list if only one element
    if (is.list(input) && length(input) == 1) input <- input[[1]]

    # Calculate net output of shape (batch_size, out_features)
    out <- self$net$forward(input)

    # Get baseline hazards
    if (use_base_hazard) {
      base_hazard <- torch::torch_tensor(self$base_hazard$hazard)$reshape(c(rep(1, out$dim()), -1))
    } else {
      base_hazard <- 1
    }

    # Add pseudo time dimension, i.e. shape (batch_size, out_features, 1)
    out <- out$unsqueeze(-1)

    # Calculate target
    if (target == "hazard") {
      out <- torch::torch_exp(out) * base_hazard
    } else if (target == "cum_hazard") {
      out <- (torch::torch_exp(out) * base_hazard)$cumsum(dim = -1)
    } else if (target == "survival") {
      out <- torch::torch_exp(-(torch::torch_exp(out) * base_hazard)$cumsum(dim = -1))
    }

    list(out$unsqueeze(-2))
  },

  # Calculate gradients of a CoxTime model
  calc_gradients = function(inputs, return_out = FALSE, use_base_hazard = TRUE,
                            target = "hazard") {

    # Clone tensor
    if (is.list(inputs)) {
      inputs <- lapply(inputs, function(i) torch::torch_clone(i))
    } else {
      inputs <- torch::torch_clone(inputs)
    }

    # Set 'requires_grad' for the input tensors
    if (is.list(inputs)) {
      lapply(inputs, function(i) i$requires_grad <- TRUE)
    } else {
      inputs$requires_grad <- TRUE
    }

    # Run input through the model
    output <- self$forward(inputs, target = "hazard",
                           use_base_hazard = FALSE)

    # Make sure output is a list
    if (!is.list(output)) output <- list(output)

    # Calculate gradients for each output layer
    grads <- lapply(output, function(out) {
      torch::autograd_grad(out$sum(), inputs)[[1]]
    })

    # Delete output if not required
    if (!return_out) output <- NULL

    list(grads = grads, outs = output)
  }
)
