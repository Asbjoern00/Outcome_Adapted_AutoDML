library(tidyverse)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

estimator_lookup <- tribble(
  ~estimator,        ~label,
  "outcome_adapted", "Outcome-adapted Neural Net",
  "0.01",            "RieszNet, Riesz Weight 0.01",
  "0.1",             "RieszNet, Riesz Weight 0.1",
  "1",               "RieszNet, Riesz Weight 1",
  "10",              "RieszNet, Riesz Weight 10",
  "100",             "RieszNet, Riesz Weight 100",
  "riesz_adapted",   "Riesz-adapted Neural Net"
)

output_path <- "plots/varying_riesz_weights_decomposition.pdf"

metric_levels <- c("bias_squared", "variance", "mse")

metric_labels <- c(
  bias_squared = "Squared Bias",
  variance = "Variance",
  mse = "MSE"
)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

read_experiment <- function(path, estimator_value = NULL, estimator_col = NULL) {
  data <- read_csv(path, show_col_types = FALSE)

  if (!is.null(estimator_value)) {
    data <- data %>%
      mutate(estimator = estimator_value)
  }

  if (!is.null(estimator_col)) {
    data <- data %>%
      mutate(estimator = as.character(.data[[estimator_col]])) %>%
      filter(estimator %in% estimator_lookup$estimator)
  }

  data %>%
    select(point_estimate, truth, estimator, seed)
}

make_metric_plot_data <- function(data) {
  bind_rows(
    data %>%
      transmute(
        estimator,
        metric = "bias_squared",
        value = bias_squared,
        se = bias_squared_se
      ),
    data %>%
      transmute(
        estimator,
        metric = "variance",
        value = variance,
        se = variance_se
      ),
    data %>%
      transmute(
        estimator,
        metric = "mse",
        value = mse,
        se = mse_se
      )
  ) %>%
    mutate(
      metric = factor(
        metric,
        levels = metric_levels,
        labels = metric_labels[metric_levels]
      ),
      lower = pmax(value - 1.96 * se, 0),
      upper = pmax(value + 1.96 * se, 0),
      estimator = factor(estimator, levels = estimator_lookup$estimator)
    ) %>%
    arrange(metric, estimator)
}

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

dope_nets <- read_experiment(
  path = "results/varying_weights_plots_experiments/ihdp_outcome_vs_riesz_adapted.csv",
  estimator_col = "training_procedure"
)

riesz_net <- read_experiment(
  path = "results/varying_weights_plots_experiments/ihdp_varying_riesz_weight.csv",
  estimator_col = "riesz_weight"
)

# -------------------------------------------------------------------------
# Summarise results
# -------------------------------------------------------------------------

summary_data <- bind_rows(
  dope_nets,
  riesz_net
) %>%
  mutate(error = point_estimate - truth) %>%
  group_by(estimator) %>%
  summarise(
    bias = mean(error),
    bias_squared = bias^2,
    variance = mean((error - bias)^2),
    mse = mean(error^2),
    bias_squared_se = 2 * abs(bias) * sd(error) / sqrt(n()),
    variance_se = sd((error - bias)^2) / sqrt(n()),
    mse_se = sd(error^2) / sqrt(n()),
    .groups = "drop"
  )

plot_data <- make_metric_plot_data(summary_data)

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

riesz_weights_decomposition <- ggplot(
  plot_data,
  aes(x = estimator, y = value, group = metric)
) +
  geom_line(linetype = "dashed") +
  geom_point() +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = 0.15
  ) +
  facet_wrap(vars(metric), nrow = 1, scales = "free_y", strip.position = "left") +
  scale_y_continuous(
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.15))
  ) +
  scale_x_discrete(
    labels = set_names(estimator_lookup$label, estimator_lookup$estimator)
  ) +
  labs(
    x = "Estimator",
    y = NULL,
    title = "RieszNet Squared Bias, Variance and MSE with Varying lambda_Riesz"
  ) +
  theme_classic(base_size = 16) +
  theme(
    axis.text.x = element_text(angle = -45, hjust = 0, vjust = 1),
    strip.background = element_blank(),
    strip.placement = "outside",
    strip.text.y.left = element_text(angle = 90),
    plot.margin = margin(t = 20, r = 60, b = 10, l = 10),
    plot.title = element_text(hjust = 0.5),
    plot.title.position = "panel"
  )

riesz_weights_decomposition

ggsave(
  filename = output_path,
  plot = riesz_weights_decomposition,
  width = 12,
  height = 6,
  create.dir = TRUE
)
